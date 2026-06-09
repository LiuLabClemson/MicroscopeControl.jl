using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.ThorCamCSC
using ImageView

using MicroscopeControl.HardwareImplementations.Meadowlark

slm = Meadowlark.MLSLM()
# Create SDK
lut_path = "C:\\Users\\nanolab\\Documents\\MeadowLark\\MeadowLark Lut file\\1024x1024_linearVoltage.lut"
Meadowlark.initializesdk()
Meadowlark.loadlut(lut_path)

isvert = true
period = 32
Meadowlark.genblazed!(slm, isvert, period, blaze_start=1, blaze_end=1024)
slm.phase = zeros(slm.height, slm.width) .+ 1.0/255
Meadowlark.writesingleimage(slm)

imshow(slm.phase)


test_cam = ThorCamCSC.ThorCamCSCCamera()

ThorCamCSC.gui(test_cam)

test_cam.exposure_time = Clonglong(10000.0) # us

data = ThorCamCSC.capture(test_cam)

ThorCamCSC.disarmcamera(test_cam)   

imshow(Float64.(data))


#Create loop to capture
sequence_frames = 5
sequence_array = Array{UInt16}(undef, test_cam.roi.height, test_cam.roi.width, sequence_frames)
for i in 1:sequence_frames
    println("Capturing frame $i of $sequence_frames")
    slm.phase = zeros(slm.height, slm.width) .+ i/255
    Meadowlark.writesingleimage(slm)
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


#Close SLM SDK
Meadowlark.closesdk()
