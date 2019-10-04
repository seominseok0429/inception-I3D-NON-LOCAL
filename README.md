# inception-I3D-NON-LOCAL

https://github.com/facebookresearch/video-nonlocal-net/blob/master/lib/models/nonlocal_helper.py <-- Original Code

https://arxiv.org/abs/1711.07971 <- Non Local Papers

https://arxiv.org/pdf/1706.03762.pdf <-Cited Papers

**The non local implementation was only a dot-product.**

![스크린샷, 2019-10-04 20-10-37](https://user-images.githubusercontent.com/33244972/66203450-5a645400-e6e3-11e9-92f5-a9fe439a18fb.png)
![스크린샷, 2019-10-04 20-11-03](https://user-images.githubusercontent.com/33244972/66203446-5801fa00-e6e3-11e9-8766-91c76140bf2f.png)

I3D resnet50 - res3's feature map size is (28,28). I3D inception - mixed_3c's feature map size is (28,28)
I3D resnet50 - res4's feature map size is (14,14). I3D inception - mixed_4d's feature map size is (14,14)

<img width="106" alt="캡처" src="https://user-images.githubusercontent.com/33244972/66208762-d913be00-e6f0-11e9-8515-c792b8b1e6db.PNG">

**Added a non-local block with matching feature map size.**

I think it's complete, but it still needs to be reviewed.

