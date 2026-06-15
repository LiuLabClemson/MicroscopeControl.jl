using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.ThorCamCSC
using ImageView

using MicroscopeControl.HardwareImplementations.Meadowlark # for the SLM
using OpenCV
const cv = OpenCV

slm = Meadowlark.MLSLM()
# Create SDK
lut_path = "C:\\Users\\nanolab\\Documents\\MeadowLark\\MeadowLark Lut file\\1024x1024_linearVoltage.lut"
Meadowlark.initializesdk() 
Meadowlark.loadlut(lut_path)

# ==============================================================================
#1. Generate 2d Matrix Grid to display on SLm
# Black (background) = 53/255, White (foreground) = 84/255, in future will want to sweep through voltages and save images at each voltage step,
# but for now just using these values to test the code
# will return tuple, the phase matrix and the coordinates of the grid needed for computing the affine transformation later. 
# ==============================================================================

phaseGrid, slm_points = genCalibrationGrid(53/255, 84/255)

println("Successfully built slm_points tensor with shape: ", size(slm_points)) # Returns (2, 1, 196)

# ==============================================================================
#2. Set the phase of the SLM to the generated grid pattern
# Write the image to the SLM. 
# ==============================================================================

slm.phase = phaseGrid
Meadowlark.writesingleimage(slm) # returns 1 for success
imshow(slm.phase')

# ==============================================================================
#3. Wait for the image to be displayed on the SLM, then trigger the camera to capture an image of the pattern displayed on the SLM. 
# Save this image as "calib_frame_v$(v).png", where $(v) is the voltage value used for that frame.
# Always disarm the camera after each capture, otherwise the camera will still be waiting for the next trigger 
#and will not capture a new image.
# ==============================================================================

test_cam = ThorCamCSC.ThorCamCSCCamera()
test_cam.exposure_time = Clonglong(79989) # 79.989 ms

if Meadowlark.imagewritecomplete(1, 5000) != 1
    error("Failed to write image to SLM within 5 seconds. Aborting capture.")
end

grid_raw = try
    ThorCamCSC.capture(test_cam)
finally
    ThorCamCSC.disarmcamera(test_cam)
end

imshow(Float64.(grid_raw)') # Transpose the image for correct orientation, since camera images are in row-major order

# ==============================================================================
#4. Would compute average of steps 1-3 for each voltage step. 
# ==============================================================================

# ==============================================================================
#5. Display uniform black (53/255) on SLM and capture image with camera. Subtract this from the (average) grid image. 
# ==============================================================================
slm.phase = fill(53/255, 1024, 1024)
Meadowlark.writesingleimage(slm)
if Meadowlark.imagewritecomplete(1, 5000) != 1
    error("Failed to write image to SLM within 5 seconds. Aborting capture.")
end
bg_raw = try
    ThorCamCSC.capture(test_cam)
finally
    ThorCamCSC.disarmcamera(test_cam)
end
imshow(Float64.(bg_raw)') # Transpose the image for correct orientation, since camera images are in row-major order


# --- subtracting out the background---
grid_float = Float64.(grid_raw)
bg_float   = Float64.(bg_raw)

diff_float = grid_float .- bg_float
diff_float[diff_float .< 0] .= 0



#this is og filitering 
max_intensity = maximum(diff_float)
if max_intensity > 0
    diff_uint8 = round.(UInt8, (diff_float ./ max_intensity) .* 255)
else
    error("Subtracted matrix is blank. Verify laser/beam path.")
end

#new filtering 
sorted_pixels = sort(vec(diff_float))
robust_max_idx = round(Int, 0.999 * length(sorted_pixels))
robust_max = sorted_pixels[robust_max_idx]

if robust_max <= 0.0
    error("Subtracted matrix contains no significant signal. Check beam path.")
end

# Clamp any stray pixels or hot spots above our robust threshold
diff_float[diff_float .> robust_max] .= robust_max

# Scale up to the absolute full 8-bit dynamic range [0, 255]
diff_uint8 = round.(UInt8, (diff_float ./ robust_max) .* 255)

imshow(Float16.(diff_uint8)') 

# --- 2. TENSOR FORMATTING ---
# Transpose from (Rows, Cols) to (X, Y) and shape into a 3D Tensor (1, W, H)
diff_uint8_3d = reshape(diff_uint8', 1, size(diff_uint8', 1), size(diff_uint8', 2)) # (..., 1, 1440, 1080)

# --- 3. SPATIAL FILTERING (Noise Suppression) ---
# Apply a light 3x3 box blur to smooth out pixel-to-pixel sensor read noise.
# This stabilizes the edge gradients for the SB corner-finding algorithm
# without washing out your large 32-pixel wide checkerboard blocks.\
#okay this works now, big diffrence was notice i need to transpose for both of the sizes!!!
diff_uint8_3d = cv.blur(diff_uint8_3d, cv.Size{Int32}(Int32(3), Int32(3)))

imshow(Float16.(dropdims(diff_uint8_3d, dims=1))') # Transpose back for correct orientation in display

#just to see if findchessboard can work with an ideal chessboard
#it does (storta, off by 2 pixels for some reason, but it does find the corners)
#control: diff_uint8 = round.(UInt8, (phaseGrid').* 255)



#imshow(Float16.(diff_uint8)') 
# Format for OpenCV: Transpose to (X,Y) and shape to 3D Tensor (1, W, H)
#control: diff_uint8_3d = reshape(diff_uint8', 1, size(diff_uint8', 1), size(diff_uint8', 2)) # (..., 1, 1080, 1440)


# ==============================================================================
#6. Use OpenCV to determine cordinates of the grid corners in the camera image, 
#and use these coordinates to compute the affine transformation between the SLM and the camera. 
#This transformation can then be used to map any point on the SLM to its corresponding point in the camera image.
# ==============================================================================
pattern_size = cv.Size{Int32}(Int32(14), Int32(14))

success, camera_points = cv.findChessboardCornersSB(diff_uint8_3d, pattern_size)

println(Array(camera_points))

if !success
    error("OpenCV failed to find corners. Check image exposure levels.")
end

println("Grid intersections successfully locked by OpenCV!")
affine_matrix, inliers = cv.estimateAffine2D(slm_points, camera_points)
    
println("\n>>> SUCCESS: 2x3 Geometric Affine Matrix Computed <<<")
display(affine_matrix)







#Close Camera
ThorCamCSC.closecamera(test_cam)

#Delete Camera SDK
ThorCamCSC.thorcamsdkuninit()

#Close SLM SDK
Meadowlark.closesdk()









#Helper Functions

#First need to find how centers of SLM map to camera, user needs to manualy change (cx, cy)
#84/255 is white, 53/255 is black, generates a black circle. 
       center_spot_mask = fill(84/255, 1024, 1024)
N = 1024
bg = 84/255
fg = 53/255
r = 12
cx = 512#514
cy = 512#560
#Note to self (512, 512)_SLM maps to (713, 435)_Camera. 
#and (514, 560)_SLM maps to (720, 540)_Camera aka the center of the camera. 
#so the scaling factor is $\frac{\sqrt{(720-713)^2 + (540-435)^2}}{\sqrt{(514-512)^2 + (560-512)^2}} \approx 2.19$, 
#which is close to the expected scaling factor of 2.0 based on the focal lengths of the camera and the SLM.
for row in 1:N, col in 1:N
    if (row - cy)^2 + (col - cx)^2 <= r^2
        center_spot_mask[col, row] = fg
    end
end
slm.phase = center_spot_mask
Meadowlark.writesingleimage(slm) # returns 1 for success


#=
Generates a symmetrical square checkerboard grid anchored to the new optical center.
Fits a 480x480 region on the SLM, mapping perfectly to the camera's 1080 height.
Returns:
- phaseGrid: 1024x1024 Float64 matrix for the SLM
- cornerMatrix: native 3D Julia array of physical corners for OpenCV calibration
=#
function genCalibrationGrid(v_dark::Float64, v_bright::Float64)
    N = 1024 #SLM Number of pixels per side
    square_size = 32
    
    # floor(1080/2.19/32) = 15, which is the maximum number of whole squares that can fit in the camera's vertical FOV
    # 15 * 32 gives a 480 pixel grid on the SLM, which maps to 480*2.19 = 1051 pixels on the camera, fitting within the 1080 pixel height.
    # 15x15 squares gives a 480x480 patch, resulting in 14x14 internal corners
    squares_per_side = 15 
    
    # Define bounds centered precisely on (514, 560)
    x_start = 274 # 514 - (15/2)*32 
    x_end   = 754  # 274 + 480
    y_start = 320 # 560 - (15/2)*32
    y_end   = 800  # 320 + 480

    # Initialize the matrix with your baseline dark voltage 
    phaseGrid = fill(v_dark, N, N)

    # Populate the checkerboard using column-major looping 
    for row in y_start:(y_end - 1)
        block_y = (row - y_start) ÷ square_size
        
        for col in x_start:(x_end - 1)
            block_x = (col - x_start) ÷ square_size
            
            # If the sum of the block coordinates is odd, write the bright voltage (84/255)
            if (block_x + block_y) % 2 == 1
                phaseGrid[col, row] = v_bright
            end
        end
    end

    # Generate the matching matrix of physical SLM internal corners
    # A 15x15 grid of blocks has exactly 14x14 internal intersections
    corners_per_side = squares_per_side - 1 # 14
    total_corners = corners_per_side * corners_per_side # 196

    # ((x, y), singleton channel dimension, corner index) to match OpenCV's expected input format
    slm_points = zeros(Float32, 2, 1, total_corners)

    # Populate the points matrix following OpenCV's scanning trajectory
    n = 1
    for r in 0:(corners_per_side-1)
        for c in 0:(corners_per_side-1)
            pt_x = x_start + (c + 1) * square_size
            pt_y = y_start + (r + 1) * square_size

            slm_points[1, 1, n] = Float32(pt_x) # X Coordinate
            slm_points[2, 1, n] = Float32(pt_y) # Y Coordinate
            n += 1
        end
    end

    return (phaseGrid, slm_points)
end
















###############test code below 

#to write the genblazed pattern.
isvert = true    
period = 32
Meadowlark.genblazed!(slm, isvert, period, blaze_start=1, blaze_end=1024) #will need to replace this with a function for a checkerboard pattern


#this will generate a circular pattern at center of slm
M1 = fill(84/255, 1024, 1024)

N = 1024
bg = 84/255
fg = 53/255
r = 12
cx = 512#514
cy = 512#560
#Note to self (512, 512)_SLM maps to (713, 435)_Camera. 
#and (514, 560)_SLM maps to (720, 540)_Camera aka the center of the camera. 
#so the scaling factor is $\frac{\sqrt{(720-713)^2 + (540-435)^2}}{\sqrt{(514-512)^2 + (560-512)^2}} \approx 2.19$, 
#which is close to the expected scaling factor of 2.0 based on the focal lengths of the camera and the SLM.
for row in 1:N, col in 1:N
    if (row - cy)^2 + (col - cx)^2 <= r^2
        M1[col, row] = fg
    end
end
slm.phase = M1

#slm.phase = zeros(slm.height, slm.width) .+ 1.0/255

#this will generate a checkerboard pattern at the center of the SLM, which can be used for calibration
#slm.phase = generate_calibration_grid(53/255)

Meadowlark.writesingleimage(slm) # returns 1 for success

imshow(slm.phase') #Transposing the phase pattern so it is displayed correctly, since the SLM uses column major order (double check this)


test_cam = ThorCamCSC.ThorCamCSCCamera()

ThorCamCSC.gui(test_cam) # Opens the camera GUI, which allows you to set the exposure time, gain, and other settings. The camera must be armed in order to capture an image, so this is a good way to set up the camera before capturing.

test_cam.exposure_time = Clonglong(10000.0) # microseconds, so this is 10 ms



data = ThorCamCSC.capture(test_cam)

ThorCamCSC.disarmcamera(test_cam) #Disarming the camera allows for the next capture to be taken, otherwise the camera is still waiting for the next trigger and will not capture a new image

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

#Delete Camera SDK
ThorCamCSC.thorcamsdkuninit()

#Close SLM SDK
Meadowlark.closesdk()


#first verion
  function generate_calibration_grid(target_voltage)
    N = 1024

    # 1. Establish your baseline values
    bg = 84 / 255
    fg = target_voltage  # This is the value you will increase in your loop

    # Initialize a flat background matrix
    M1 = fill(bg, N, N)

    # 2. Define the grid boundaries anchored to your empirical center (549, 555)
    square_size = 32

    x_start = 229
    x_end = 869
    y_start = 331
    y_end = 779

    # 3. Loop through the matrix using Julia's column-major indexing efficiency
    # external loop over rows (Y), internal loop over columns (X)
    for row in y_start:y_end
        # Calculate which horizontal block row we are in
        block_row = (row - y_start) ÷ square_size

        for col in x_start:x_end
            # Calculate which vertical block column we are in
            block_col = (col - x_start) ÷ square_size

            # Check if the sum of block coordinates is odd or even to alternate colors
            if (block_row + block_col) % 2 == 1
                M1[col, row] = fg
            end
        end
    end

    return M1
end

#updated version
function genCalibrationGrid(target_voltage)
    N = 1024

    # --- A. Define Known SLM Corners ---
    square_size = 32
    corners_x = 19 # Number of INTERNAL corners horizontally
    corners_y = 13 # Number of INTERNAL corners vertically

    # Starting coordinate of the first INTERNAL corner
    # (x_start of your grid + 1 square size)
    start_x = 229 + square_size
    start_y = 331 + square_size

    # Build the array of known SLM corner points
    slm_points = []
    for row in 0:(corners_y-1)
        for col in 0:(corners_x-1)
            pt_x = start_x + (col * square_size)
            pt_y = start_y + (row * square_size)
            push!(slm_points, cv.Point2f(Float32(pt_x), Float32(pt_y)))
        end
    end
    # slm_points is now a flat list matching OpenCV's output order
end



using Dates # Built-in library for handling time delays

function run_voltage_calibration_sweep(slm::MLSLM)
    board_number = 1
    timeout_ms = 5000 # 5-second safety timeout for DMA transfers
    
    # Define your voltage sweep range (e.g., from 54 to 114 in steps of 2)
    voltage_steps = 54:2:114
    
    println("Starting calibration sweep...")
    
    for v in voltage_steps
        # 1. Generate your 2D grid matrix (returns floats)
        slm.phase = generate_calibration_grid(v / 255)
        
        
        success_write = Meadowlark.writesingleimage(slm)
        
        if success_write != 1
            error("DMA Image Write failed at voltage step $v")
        end
        
        # 4. CRITICAL: Wait for the hardware memory bank to clear before proceeding
        #success_complete = Meadowlark.ImageWriteComplete(board_number, timeout_ms)
        #if success_complete != 1
        #    error("Hardware bank failed to clear at voltage step $v")
        #end
        
        println("SLM successfully displaying checkerboard at voltage: $v/255")
        
        # ================================================================
        # YOUR CAMERA CODE GOES HERE
        # This is where you trigger ThorImageCAM to take a snapshot 
        # and save it as "calib_frame_v$(v).png"
        # ================================================================
        
        # 5. Software Wait: Hold the image on screen for 2.0 seconds
        sleep(2.0) 
    end
    
    println("Sweep complete! All calibration frames projected.")
end

run_voltage_calibration_sweep(slm)