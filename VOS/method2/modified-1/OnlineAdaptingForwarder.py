from Forwarding.OneshotForwarder import OneshotForwarder
from datasets.Util.Timer import Timer
from Measures import average_measures
from Log import log
import Measures
import cv2

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion

VOID_LABEL = 255


class OnlineAdaptingForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(OnlineAdaptingForwarder, self).__init__(engine)
    self.n_adaptation_steps = self.config.int("n_adaptation_steps", 36)	#
    self.adaptation_interval = self.config.int("adaptation_interval", 3)
    self.adaptation_learning_rate = self.config.float("adaptation_learning_rate")
    self.posterior_positive_threshold = self.config.float("posterior_positive_threshold", 0.97)
    self.distance_negative_threshold = self.config.float("distance_negative_threshold", 150.0)
    self.adaptation_loss_scale = self.config.float("adaptation_loss_scale", 0.1)
    self.debug = self.config.bool("adapt_debug", False)
    self.erosion_size = self.config.int("adaptation_erosion_size", 20)
    self.use_positives = self.config.bool("use_positives", True)
    self.use_negatives = self.config.bool("use_negatives", True)

  def _oneshot_forward_video(self, video_idx, save_logits):
    with Timer():
      # finetune on first frame
      print("--------------------------------This mine change-------------------------")
      print("--------OnlineAdaptingForwarder finetune ---------------")
      self._finetune(video_idx, n_finetune_steps=self.n_finetune_steps)

      network = self.engine.test_network
      targets = network.raw_labels
      ys = network.y_softmax
      ys = self._adjust_results_to_targets(ys, targets)
      data = self.val_data

      n, measures, ys_argmax_val, logits_val, targets_val = self._process_forward_minibatch(
        data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=0)
      assert n == 1
      
      n_frames = data.num_examples_per_epoch()
      print("-------------n_frames: ",n_frames)
      measures_video = []

      last_mask = targets_val[0]
      f=open("/export/zhu/OnAVOS/Forwarding/kframe/kfrme_"+str(self.train_data.video_tag(video_idx))+".txt","w")

      borderFrameLst=self.extractBorderFrame_list(video_idx)
      for m in borderFrameLst:
        print m
        f.write(str(m)+'\n')
      f.close()
      print(borderFrameLst)
      for t in xrange(1, n_frames):
        print("frameth:",t)
        def get_posteriors():
          n_, _, _, logits_val_, _ = self._process_forward_minibatch(
              data, network, save_logits=False, save_results=False, targets=targets, ys=ys, start_frame_idx=t)
          assert n_ == 1
          return logits_val_[0]

        # recover annotation data with results
        pFrameborderFrameLst=[]
        for i in borderFrameLst:
          if  i<t:
            pFrameborderFrameLst.append(i)

        # forward current frame using adapted model
        
        negatives = self._adapt(video_idx, t, last_mask, get_posteriors,pFrameborderFrameLst)
        n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(
            data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
        
        self.train_data._videos[video_idx][t]["label"]=ys_argmax_val[0]
       

        assert n == 1
        assert len(measures) == 1
        measure = measures[0]
        print >> log.v5, "frame", t, ":", measure
        measures_video.append(measure)
        #print("measures_video",measures_video)
        last_mask = ys_argmax_val[0]

        # prune negatives from last mask
        # negatives are None if we think that the target is lost
        if negatives is not None and self.use_negatives:
          last_mask[negatives] = 0

      measures_video[:-1] = measures_video[:-1]
      measures_video = average_measures(measures_video)
      print >> log.v1, "sequence", video_idx + 1, data.video_tag(video_idx), measures_video

  def getBoundingBox(self,imageMask):
    w,h,c =imageMask.shape
    print("w=%d,h=%d,c=%d"%(w,h,c))
    if c>2:
        gray = cv2.cvtColor(imageMask, cv2.COLOR_BGR2GRAY)
    else:
       gray = np.multiply(imageMask,255)
    ret,thresh = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
    _,contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_ret=[0, 0 ,gray.shape[0] ,gray.shape[1]]
    rect=rect_ret
    #cv2.imwrite('label.png',imageMask)
    for c in contours:
          print(cv2.contourArea(c))/(gray.shape[0]*gray.shape[1])
          if cv2.contourArea(c)/(gray.shape[0]*gray.shape[1]) <0.01 :
              continue
          rect = cv2.boundingRect(c)

    rect_ret.append(max(int(rect[0]-rect[2]*0.1),0))
    rect_ret.append(max(int(rect[1]-rect[3]*0.1),0))
    rect_ret.append(min(int(rect[2]*1.2),gray.shape[1]))
    rect_ret.append(min(int(rect[3]*1.2),gray.shape[0]))
      #print(rect_ret)
    return rect_ret


  def extractBorderFrame_list(self, video_idx):
    list_ret = [0]
    leng = len(self.train_data._videos[video_idx])
    rect = self.getBoundingBox(self.train_data._videos[video_idx][0]['label'])
    diff_list = []
    for i in range(1,leng):
      pFrame = self.train_data._videos[video_idx][i]["unnormalized_img"][rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
      ppFrame=self.train_data._videos[video_idx][i-1]["unnormalized_img"][rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
      absDiff = sum(sum(sum(np.square(abs(pFrame-ppFrame)))))
      diff_list.append(absDiff)
    meanDiff = np.mean(diff_list)
    stdDiff = np.sqrt(np.var(diff_list))
    threshold = meanDiff+stdDiff
    for i in range(len(diff_list)):
      if diff_list[i] > threshold:
        list_ret.append(i)

    return list_ret

 

  def _adapt(self, video_idx, frame_idx, last_mask, get_posteriors_fn,kframe_lst): 
    eroded_mask = grey_erosion(last_mask, size=(self.erosion_size, self.erosion_size, 1))
    adaptation_target2=last_mask
    adaptation_target = np.zeros_like(last_mask)
    adaptation_target[:] = VOID_LABEL

    current_posteriors = get_posteriors_fn()
    positives = current_posteriors[:, :, 1] > self.posterior_positive_threshold
    if self.use_positives:
      adaptation_target[positives] = 1

    dt = distance_transform_edt(np.logical_not(eroded_mask))
    threshold = self.distance_negative_threshold
    negatives = dt > threshold
    if self.use_negatives:
      adaptation_target[negatives] = 0

    do_adaptation = eroded_mask.sum() > 0
    print('frame_idx',frame_idx)
    #if self.debug:
    #  adaptation_target_visualization = adaptation_target.copy()
    #  adaptation_target_visualization[adaptation_target == 1] = 128
    #  if not do_adaptation:
    #    adaptation_target_visualization[:] = VOID_LABEL
    #  from scipy.misc import imsave
    #  folder = self.val_data.video_tag().replace("__", "/")
    #  imsave("forwarded/" + self.model + "/valid/" + folder + "/adaptation_%05d.png" % frame_idx,
    #         np.squeeze(adaptation_target_visualization))

    self.train_data.set_video_idx(video_idx)
    threshold_=0.05
     
    #for idx in xrange(self.n_adaptation_steps):    #
    for idx in xrange(frame_idx): 
      do_step = True
      #print(kframe_lst)
      if idx % len(kframe_lst) == 0:	#adaptation_interval
        if do_adaptation:
          #print("NewIter")
          #print("idx % self.adaptation_interval == 0",idx % self.adaptation_interval == 0)
          feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
          feed_dict[self.train_data.get_label_placeholder()] = adaptation_target             #
          loss_scale = self.adaptation_loss_scale*5
          adaption_frame_idx = frame_idx
        else:
          print >> log.v4, "skipping current frame adaptation, since the target seems to be lost"
          do_step = False

      elif idx % len(kframe_lst) == 1:
        if len(kframe_lst)==2:
          feed_dict=self.train_data.feed_dict_for_video_frame(kframe_lst[1],with_annotations=True)
          loss_scale = self.adaptation_loss_scale*10
          adaption_frame_idx=kframe_lst[1]
        else:
          feed_dict=self.train_data.feed_dict_for_video_frame(kframe_lst[-1],with_annotations=True)
          loss_scale = self.adaptation_loss_scale*10
          adaption_frame_idx=kframe_lst[-1]
        

      elif idx % len(kframe_lst)==2:
        #print "----------------2------------"
        #print "len kframe",len(kframe_lst)
        if len(kframe_lst)>2:
           feed_dict=self.train_data.feed_dict_for_video_frame(kframe_lst[-2],with_annotations=True)
           loss_scale=self.adaptation_loss_scale*10
           adaption_frame_idx=kframe_lst[-2]
      else:
        # mix in first frame to avoid drift
        # (do this even if we think the target is lost, since then this can help to find back the target)
        feed_dict = self.train_data.feed_dict_for_video_frame(0, with_annotations=True)
        loss_scale = 1.0
        adaption_frame_idx = 0        


      if do_step:
        #self._finetune(video_idx, n_finetune_steps=5)
        loss, measures, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale,learning_rate=self.adaptation_learning_rate)
        #iou=Measures.calc_iou(measures,n_imgs,[0])
        assert n_imgs == 1
        #print >> log.v4, "adapting on frame", adaption_frame_idx, "of sequence", video_idx + 1, \
        #    self.train_data.video_tag(video_idx), "loss:", loss,"iou:",iou
    if do_adaptation:
      return negatives
    else:
      return None
