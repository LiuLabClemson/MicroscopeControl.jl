using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.ThorCamCSC
using ImageView




test_cam = ThorCamCSC.ThorCamCSCCamera()

ThorCamCSC.gui(test_cam)

test_cam.exposure_time = Clonglong(10000.0) # us

data = ThorCamCSC.capture(test_cam)

ThorCamCSC.disarmcamera(test_cam)   


#Create loop to capture
sequence_frames = 5
sequence_array = Array{UInt16}(undef, test_cam.roi.height, test_cam.roi.width, sequence_frames)
for i in 1:sequence_frames
    println("Capturing frame $i of $sequence_frames")
    data = ThorCamCSC.capture(test_cam)

    ThorCamCSC.disarmcamera(test_cam)   

    sequence_array[:, :, i] = data
end

imgs = Float64.(sequence_array)

imshow(imgs)





#Close Camera
ThorCamCSC.closecamera(test_cam)

#Delete SDK
ThorCamCSC.thorcamsdkuninit()



