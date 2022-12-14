## Below is the model architecture summary of the Inflated 3d inception model.

```sh
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1      [-1, 64, 7, 128, 128]          65,856
       BatchNorm3d-2      [-1, 64, 7, 128, 128]             128
            Unit3D-3      [-1, 64, 7, 128, 128]               0
MaxPool3dSamePadding-4        [-1, 64, 7, 64, 64]               0
            Conv3d-5        [-1, 64, 7, 64, 64]           4,096
       BatchNorm3d-6        [-1, 64, 7, 64, 64]             128
            Unit3D-7        [-1, 64, 7, 64, 64]               0
            Conv3d-8       [-1, 192, 7, 64, 64]         331,776
       BatchNorm3d-9       [-1, 192, 7, 64, 64]             384
           Unit3D-10       [-1, 192, 7, 64, 64]               0
MaxPool3dSamePadding-11       [-1, 192, 7, 32, 32]               0
           Conv3d-12        [-1, 64, 7, 32, 32]          12,288
      BatchNorm3d-13        [-1, 64, 7, 32, 32]             128
           Unit3D-14        [-1, 64, 7, 32, 32]               0
           Conv3d-15        [-1, 96, 7, 32, 32]          18,432
      BatchNorm3d-16        [-1, 96, 7, 32, 32]             192
           Unit3D-17        [-1, 96, 7, 32, 32]               0
           Conv3d-18       [-1, 128, 7, 32, 32]         331,776
      BatchNorm3d-19       [-1, 128, 7, 32, 32]             256
           Unit3D-20       [-1, 128, 7, 32, 32]               0
           Conv3d-21        [-1, 16, 7, 32, 32]           3,072
      BatchNorm3d-22        [-1, 16, 7, 32, 32]              32
           Unit3D-23        [-1, 16, 7, 32, 32]               0
           Conv3d-24        [-1, 32, 7, 32, 32]          13,824
      BatchNorm3d-25        [-1, 32, 7, 32, 32]              64
           Unit3D-26        [-1, 32, 7, 32, 32]               0
MaxPool3dSamePadding-27       [-1, 192, 7, 32, 32]               0
           Conv3d-28        [-1, 32, 7, 32, 32]           6,144
      BatchNorm3d-29        [-1, 32, 7, 32, 32]              64
           Unit3D-30        [-1, 32, 7, 32, 32]               0
  InceptionModule-31       [-1, 256, 7, 32, 32]               0
           Conv3d-32       [-1, 128, 7, 32, 32]          32,768
      BatchNorm3d-33       [-1, 128, 7, 32, 32]             256
           Unit3D-34       [-1, 128, 7, 32, 32]               0
           Conv3d-35       [-1, 128, 7, 32, 32]          32,768
      BatchNorm3d-36       [-1, 128, 7, 32, 32]             256
           Unit3D-37       [-1, 128, 7, 32, 32]               0
           Conv3d-38       [-1, 192, 7, 32, 32]         663,552
      BatchNorm3d-39       [-1, 192, 7, 32, 32]             384
           Unit3D-40       [-1, 192, 7, 32, 32]               0
           Conv3d-41        [-1, 32, 7, 32, 32]           8,192
      BatchNorm3d-42        [-1, 32, 7, 32, 32]              64
           Unit3D-43        [-1, 32, 7, 32, 32]               0
           Conv3d-44        [-1, 96, 7, 32, 32]          82,944
      BatchNorm3d-45        [-1, 96, 7, 32, 32]             192
           Unit3D-46        [-1, 96, 7, 32, 32]               0
MaxPool3dSamePadding-47       [-1, 256, 7, 32, 32]               0
           Conv3d-48        [-1, 64, 7, 32, 32]          16,384
      BatchNorm3d-49        [-1, 64, 7, 32, 32]             128
           Unit3D-50        [-1, 64, 7, 32, 32]               0
  InceptionModule-51       [-1, 480, 7, 32, 32]               0
MaxPool3dSamePadding-52       [-1, 480, 4, 16, 16]               0
           Conv3d-53       [-1, 192, 4, 16, 16]          92,160
      BatchNorm3d-54       [-1, 192, 4, 16, 16]             384
           Unit3D-55       [-1, 192, 4, 16, 16]               0
           Conv3d-56        [-1, 96, 4, 16, 16]          46,080
      BatchNorm3d-57        [-1, 96, 4, 16, 16]             192
           Unit3D-58        [-1, 96, 4, 16, 16]               0
           Conv3d-59       [-1, 208, 4, 16, 16]         539,136
      BatchNorm3d-60       [-1, 208, 4, 16, 16]             416
           Unit3D-61       [-1, 208, 4, 16, 16]               0
           Conv3d-62        [-1, 16, 4, 16, 16]           7,680
      BatchNorm3d-63        [-1, 16, 4, 16, 16]              32
           Unit3D-64        [-1, 16, 4, 16, 16]               0
           Conv3d-65        [-1, 48, 4, 16, 16]          20,736
      BatchNorm3d-66        [-1, 48, 4, 16, 16]              96
           Unit3D-67        [-1, 48, 4, 16, 16]               0
MaxPool3dSamePadding-68       [-1, 480, 4, 16, 16]               0
           Conv3d-69        [-1, 64, 4, 16, 16]          30,720
      BatchNorm3d-70        [-1, 64, 4, 16, 16]             128
           Unit3D-71        [-1, 64, 4, 16, 16]               0
  InceptionModule-72       [-1, 512, 4, 16, 16]               0
           Conv3d-73       [-1, 160, 4, 16, 16]          81,920
      BatchNorm3d-74       [-1, 160, 4, 16, 16]             320
           Unit3D-75       [-1, 160, 4, 16, 16]               0
           Conv3d-76       [-1, 112, 4, 16, 16]          57,344
      BatchNorm3d-77       [-1, 112, 4, 16, 16]             224
           Unit3D-78       [-1, 112, 4, 16, 16]               0
           Conv3d-79       [-1, 224, 4, 16, 16]         677,376
      BatchNorm3d-80       [-1, 224, 4, 16, 16]             448
           Unit3D-81       [-1, 224, 4, 16, 16]               0
           Conv3d-82        [-1, 24, 4, 16, 16]          12,288
      BatchNorm3d-83        [-1, 24, 4, 16, 16]              48
           Unit3D-84        [-1, 24, 4, 16, 16]               0
           Conv3d-85        [-1, 64, 4, 16, 16]          41,472
      BatchNorm3d-86        [-1, 64, 4, 16, 16]             128
           Unit3D-87        [-1, 64, 4, 16, 16]               0
MaxPool3dSamePadding-88       [-1, 512, 4, 16, 16]               0
           Conv3d-89        [-1, 64, 4, 16, 16]          32,768
      BatchNorm3d-90        [-1, 64, 4, 16, 16]             128
           Unit3D-91        [-1, 64, 4, 16, 16]               0
  InceptionModule-92       [-1, 512, 4, 16, 16]               0
           Conv3d-93       [-1, 128, 4, 16, 16]          65,536
      BatchNorm3d-94       [-1, 128, 4, 16, 16]             256
           Unit3D-95       [-1, 128, 4, 16, 16]               0
           Conv3d-96       [-1, 128, 4, 16, 16]          65,536
      BatchNorm3d-97       [-1, 128, 4, 16, 16]             256
           Unit3D-98       [-1, 128, 4, 16, 16]               0
           Conv3d-99       [-1, 256, 4, 16, 16]         884,736
     BatchNorm3d-100       [-1, 256, 4, 16, 16]             512
          Unit3D-101       [-1, 256, 4, 16, 16]               0
          Conv3d-102        [-1, 24, 4, 16, 16]          12,288
     BatchNorm3d-103        [-1, 24, 4, 16, 16]              48
          Unit3D-104        [-1, 24, 4, 16, 16]               0
          Conv3d-105        [-1, 64, 4, 16, 16]          41,472
     BatchNorm3d-106        [-1, 64, 4, 16, 16]             128
          Unit3D-107        [-1, 64, 4, 16, 16]               0
MaxPool3dSamePadding-108       [-1, 512, 4, 16, 16]               0
          Conv3d-109        [-1, 64, 4, 16, 16]          32,768
     BatchNorm3d-110        [-1, 64, 4, 16, 16]             128
          Unit3D-111        [-1, 64, 4, 16, 16]               0
 InceptionModule-112       [-1, 512, 4, 16, 16]               0
          Conv3d-113       [-1, 112, 4, 16, 16]          57,344
     BatchNorm3d-114       [-1, 112, 4, 16, 16]             224
          Unit3D-115       [-1, 112, 4, 16, 16]               0
          Conv3d-116       [-1, 144, 4, 16, 16]          73,728
     BatchNorm3d-117       [-1, 144, 4, 16, 16]             288
          Unit3D-118       [-1, 144, 4, 16, 16]               0
          Conv3d-119       [-1, 288, 4, 16, 16]       1,119,744
     BatchNorm3d-120       [-1, 288, 4, 16, 16]             576
          Unit3D-121       [-1, 288, 4, 16, 16]               0
          Conv3d-122        [-1, 32, 4, 16, 16]          16,384
     BatchNorm3d-123        [-1, 32, 4, 16, 16]              64
          Unit3D-124        [-1, 32, 4, 16, 16]               0
          Conv3d-125        [-1, 64, 4, 16, 16]          55,296
     BatchNorm3d-126        [-1, 64, 4, 16, 16]             128
          Unit3D-127        [-1, 64, 4, 16, 16]               0
MaxPool3dSamePadding-128       [-1, 512, 4, 16, 16]               0
          Conv3d-129        [-1, 64, 4, 16, 16]          32,768
     BatchNorm3d-130        [-1, 64, 4, 16, 16]             128
          Unit3D-131        [-1, 64, 4, 16, 16]               0
 InceptionModule-132       [-1, 528, 4, 16, 16]               0
          Conv3d-133       [-1, 256, 4, 16, 16]         135,168
     BatchNorm3d-134       [-1, 256, 4, 16, 16]             512
          Unit3D-135       [-1, 256, 4, 16, 16]               0
          Conv3d-136       [-1, 160, 4, 16, 16]          84,480
     BatchNorm3d-137       [-1, 160, 4, 16, 16]             320
          Unit3D-138       [-1, 160, 4, 16, 16]               0
          Conv3d-139       [-1, 320, 4, 16, 16]       1,382,400
     BatchNorm3d-140       [-1, 320, 4, 16, 16]             640
          Unit3D-141       [-1, 320, 4, 16, 16]               0
          Conv3d-142        [-1, 32, 4, 16, 16]          16,896
     BatchNorm3d-143        [-1, 32, 4, 16, 16]              64
          Unit3D-144        [-1, 32, 4, 16, 16]               0
          Conv3d-145       [-1, 128, 4, 16, 16]         110,592
     BatchNorm3d-146       [-1, 128, 4, 16, 16]             256
          Unit3D-147       [-1, 128, 4, 16, 16]               0
MaxPool3dSamePadding-148       [-1, 528, 4, 16, 16]               0
          Conv3d-149       [-1, 128, 4, 16, 16]          67,584
     BatchNorm3d-150       [-1, 128, 4, 16, 16]             256
          Unit3D-151       [-1, 128, 4, 16, 16]               0
 InceptionModule-152       [-1, 832, 4, 16, 16]               0
MaxPool3dSamePadding-153         [-1, 832, 2, 8, 8]               0
          Conv3d-154         [-1, 256, 2, 8, 8]         212,992
     BatchNorm3d-155         [-1, 256, 2, 8, 8]             512
          Unit3D-156         [-1, 256, 2, 8, 8]               0
          Conv3d-157         [-1, 160, 2, 8, 8]         133,120
     BatchNorm3d-158         [-1, 160, 2, 8, 8]             320
          Unit3D-159         [-1, 160, 2, 8, 8]               0
          Conv3d-160         [-1, 320, 2, 8, 8]       1,382,400
     BatchNorm3d-161         [-1, 320, 2, 8, 8]             640
          Unit3D-162         [-1, 320, 2, 8, 8]               0
          Conv3d-163          [-1, 32, 2, 8, 8]          26,624
     BatchNorm3d-164          [-1, 32, 2, 8, 8]              64
          Unit3D-165          [-1, 32, 2, 8, 8]               0
          Conv3d-166         [-1, 128, 2, 8, 8]         110,592
     BatchNorm3d-167         [-1, 128, 2, 8, 8]             256
          Unit3D-168         [-1, 128, 2, 8, 8]               0
MaxPool3dSamePadding-169         [-1, 832, 2, 8, 8]               0
          Conv3d-170         [-1, 128, 2, 8, 8]         106,496
     BatchNorm3d-171         [-1, 128, 2, 8, 8]             256
          Unit3D-172         [-1, 128, 2, 8, 8]               0
 InceptionModule-173         [-1, 832, 2, 8, 8]               0
          Conv3d-174         [-1, 384, 2, 8, 8]         319,488
     BatchNorm3d-175         [-1, 384, 2, 8, 8]             768
          Unit3D-176         [-1, 384, 2, 8, 8]               0
          Conv3d-177         [-1, 192, 2, 8, 8]         159,744
     BatchNorm3d-178         [-1, 192, 2, 8, 8]             384
          Unit3D-179         [-1, 192, 2, 8, 8]               0
          Conv3d-180         [-1, 384, 2, 8, 8]       1,990,656
     BatchNorm3d-181         [-1, 384, 2, 8, 8]             768
          Unit3D-182         [-1, 384, 2, 8, 8]               0
          Conv3d-183          [-1, 48, 2, 8, 8]          39,936
     BatchNorm3d-184          [-1, 48, 2, 8, 8]              96
          Unit3D-185          [-1, 48, 2, 8, 8]               0
          Conv3d-186         [-1, 128, 2, 8, 8]         165,888
     BatchNorm3d-187         [-1, 128, 2, 8, 8]             256
          Unit3D-188         [-1, 128, 2, 8, 8]               0
MaxPool3dSamePadding-189         [-1, 832, 2, 8, 8]               0
          Conv3d-190         [-1, 128, 2, 8, 8]         106,496
     BatchNorm3d-191         [-1, 128, 2, 8, 8]             256
          Unit3D-192         [-1, 128, 2, 8, 8]               0
 InceptionModule-193        [-1, 1024, 2, 8, 8]               0
       AvgPool3d-194        [-1, 1024, 1, 2, 2]               0
         Dropout-195        [-1, 1024, 1, 2, 2]               0
          Conv3d-196         [-1, 400, 1, 2, 2]         410,000
          Unit3D-197         [-1, 400, 1, 2, 2]               0
================================================================
Total params: 12,697,264
Trainable params: 12,697,264
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 10.50
Forward/backward pass size (MB): 733.79
Params size (MB): 48.44
Estimated Total Size (MB): 792.73
----------------------------------------------------------------
None

```
