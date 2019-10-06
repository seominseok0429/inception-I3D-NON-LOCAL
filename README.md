# inception-I3D-NON-LOCAL

https://github.com/facebookresearch/video-nonlocal-net/blob/master/lib/models/nonlocal_helper.py <-- Original Code

https://arxiv.org/abs/1711.07971 <- Non Local Papers

https://arxiv.org/pdf/1706.03762.pdf <-Cited Papers


https://github.com/kenshohara/3D-ResNets-PyTorch <- base code
https://github.com/hassony2/kinetics_i3d_pytorch <- i3d code
https://github.com/Tushar-N/pytorch-resnet3d <- non local code

**The non local implementation was only a dot-product.**

![스크린샷, 2019-10-04 20-10-37](https://user-images.githubusercontent.com/33244972/66203450-5a645400-e6e3-11e9-92f5-a9fe439a18fb.png)
![스크린샷, 2019-10-04 20-11-03](https://user-images.githubusercontent.com/33244972/66203446-5801fa00-e6e3-11e9-8766-91c76140bf2f.png)
<img width="200" alt="캡처" src="https://user-images.githubusercontent.com/33244972/66208762-d913be00-e6f0-11e9-8515-c792b8b1e6db.PNG">

***

**Added a non-local block with matching feature map size.**

I3D resnet50 - res3's feature map size is (28,28). I3D inception - mixed_3c's feature map size is (28,28)

I3D resnet50 - res4's feature map size is (14,14). I3D inception - mixed_4d's feature map size is (14,14)



***

**If tf_begin_index is 8, the front of the added NonLocalBlock is all freezed and finetunes the rest.**
**if only_nonlocal True,  Freeze everything except the non-local block and the last layer.**


python3 main.py --root ./ \
        --video_path optical_flow_tvl1 \
        --annotation_path testTrainMulti_7030_splits/hmdb51_1.json \
        --result_path results \
        --dataset hmdb51_optical_flow_faster \
        --n_classes 400 --n_finetune_classes 51 \
        --model i3d --batch_size 8 \
        --n_threads 20 --checkpoint 1 --lr_patience 2 --epoch_multi 10 \
        --learning_rate 0.001 --sample_duration 64 \
        --sample_size 224 --non_local True \
        --dropout_prob 0.5 --pretrain_path pretrained/model_flow.pth \
        --ft_begin_index 8 --non_local True
