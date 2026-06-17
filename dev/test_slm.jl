using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.ThorCamCSC
using ImageView
using Images

using MicroscopeControl.HardwareImplementations.Meadowlark # for the SLM
using OpenCV
const cv = OpenCV

#for testing purposes
using CairoMakie
CM = CairoMakie
using Statistics


#Constants
const slm_square_size = 32 
const optical_scaling = 2.268 #this number was determined empircally with the helper functions



# Set Up SLM
slm = Meadowlark.MLSLM()
lut_path = "C:\\Users\\nanolab\\Documents\\MeadowLark\\MeadowLark Lut file\\1024x1024_linearVoltage.lut"
Meadowlark.initializesdk() 
Meadowlark.loadlut(lut_path)

# ==============================================================================
#1. Generate 2d Matrix Grid to display on SLM
# Black (background) = 53/255, White (foreground) = 84/255, in future will want to sweep through voltages and save images at each voltage step,
# but for now just using these values to test the code
# will return tuple, the phase matrix and the coordinates of the grid needed for computing the affine transformation later. 
# ==============================================================================

phaseGrid, slm_points = genCalibrationGrid(53/255, 84/255)
println("Successfully built slm_points tensor with shape: ", size(slm_points)) # Returns (2, 1, 84)

# ==============================================================================
#2. Set the phase of the SLM to the generated grid pattern
# Write the image to the SLM. 
# ==============================================================================

slm.phase = phaseGrid
Meadowlark.writesingleimage(slm) # returns 1 for success
imshow(slm.phase')


# ==============================================================================
#3. Setup camera and trigger it to capture an image of the pattern displayed on the SLM. 
# In future, may save this image as "calib_frame_v$(v).png", where $(v) is the voltage value used for that frame.
# Always disarm the camera after each capture, otherwise the camera will still be waiting for the next trigger 
#and will not capture a new image.
# ==============================================================================

test_cam = ThorCamCSC.ThorCamCSCCamera()
test_cam.exposure_time = Clonglong(38897) # 38.897 ms, emprically determined 


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
# Determine the centers of the White Squares --> camera_points
# ==============================================================================
slm.phase = fill(53/255, 1024, 1024)
Meadowlark.writesingleimage(slm)

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

sorted_pixels = sort(vec(diff_float))
robust_max_idx = round(Int, 0.999 * length(sorted_pixels))
robust_max = sorted_pixels[robust_max_idx]

if robust_max <= 0.0
    error("Subtracted matrix contains no significant signal. Check beam path.")
end

# Clamp any stray pixels or hot spots above our robust threshold
diff_float[diff_float .> robust_max] .= robust_max

# Scale  to [0.0, 1.0] Float64
norm_gray = diff_float ./ robust_max

# The box filter kernel size must be an odd integer to have a perfect center pixel
ideal_camera_width = slm_square_size * optical_scaling # ~70
grid_size = round(Int, ideal_camera_width)
if grid_size % 2 == 0
    grid_size += 1 # Force it to be odd (e.g., 71 pixels)
end

# Apply the normalized box filter
h1 = ones(grid_size, grid_size)
img_filtered = imfilter(norm_gray, centered(h1))

# Locate the sharp peak centers
localmax = mapwindow(maximum, img_filtered, (grid_size, grid_size)) .== img_filtered
img_peaks = localmax .* img_filtered

# Since the peaks are now sharp pyramids scaled to [0.0, 1.0], 
# a fractional threshold easily isolates the true centers.
threshold = 50 #Determined Empirically
pts = findall(img_peaks .> threshold)
pts = filter(p -> (p[1] > slm_square_size && p[1] < (1080 - slm_square_size) && 
                   p[2] > slm_square_size && p[2] < (1440 - slm_square_size)), pts)

# Group by Row index (p[1]), then sort by Column index (p[2])
y_tolerance = 15
unique_y = sort(unique([p[1] for p in pts])) 
y_groups = []
for y_val in unique_y
    if isempty(y_groups) || (y_val - y_groups[end][1] > y_tolerance)
        push!(y_groups, [y_val])
    else
        push!(y_groups[end], y_val)
    end
end

x_coords = Float32[]
y_coords = Float32[]
for y_group in y_groups
    pts_in_group = filter(p -> p[1] in y_group, pts)
    pts_sorted = sort(pts_in_group, by=p -> p[2]) # Sort Left-to-Right
    for p in pts_sorted
        push!(x_coords, Float32(p[2])) # True Space X
        push!(y_coords, Float32(p[1])) # True Space Y
    end
end


# Force Makie to display standard Top-Left Image Space
fig = Figure(size=(800, 600))
# yreversed=true puts 0 at the top, matching standard camera coordinate space
ax = CM.Axis(fig[1, 1], title="Grid Center Detection (True Top-Left Space)", yreversed=true)

# Transposing norm_gray ensures rows map to the Y axis and columns map to X
heatmap!(ax, norm_gray', colormap=:grays, colorrange=(0, 0.3))
scatter!(ax, x_coords, y_coords, color=:transparent, markersize=10, strokecolor=:red, strokewidth=2)
#= #To visulize order of coordinates
labels_vec = string.(1:length(x_coords))

text!(ax, x_coords, y_coords, 
      text = labels_vec, 
      color = :yellow, 
      fontsize = 14, 
      font = :bold,
      align = (:center, :center)) # Centers the text directly on top of the scatter point
=#
display(fig)


# Estimate grid size from horizontal spacing changes
x_diff = diff(x_coords)
grid_size_est = Statistics.mean(x_diff[x_diff .> 20]) #~ 64*2.16=138.24

# Convert to 3d tensor of coordinates
total_points = length(x_coords)
camera_points = zeros(Float32, 2, 1, total_points)

for i in 1:total_points
    camera_points[1, 1, i] = x_coords[i] 
    camera_points[2, 1, i] = y_coords[i] 
end


# ==============================================================================
#6. Compute affine matrix
# ==============================================================================
# 
if size(camera_points, 3) != size(slm_points, 3)
    error("Matrix size mismatch! Camera found $(size(camera_points,3)) peaks, but SLM expects $(size(slm_points,3)) centers.")
else
    affine_matrix, inliers = cv.estimateAffine2D(
        slm_points,
        camera_points;
        method=Int64(cv.RANSAC),
        ransacReprojThreshold=3.0,
        maxIters=UInt64(2000),
        confidence=0.99,
        refineIters=UInt64(10)
    )
    println("Affine transformation matrix successfully calculated!")
end


errors = Float64[]

for i in 1:size(slm_points, 3)
    xs = slm_points[1,1,i]
    ys = slm_points[2,1,i]

    xc_pred = affine_matrix[1,1,1]*xs +
              affine_matrix[1,2,1]*ys +
              affine_matrix[1,3,1]

    yc_pred = affine_matrix[1,1,2]*xs +
              affine_matrix[1,2,2]*ys +
              affine_matrix[1,3,2]

    xc_obs = camera_points[1,1,i]
    yc_obs = camera_points[2,1,i]

    push!(errors,
          hypot(xc_pred - xc_obs,
                yc_pred - yc_obs))
end

println("Mean error = ", mean(errors)) #ideally should be less than 1-2 px
println("Max error  = ", maximum(errors)) # ideally less than 5px
println("Number of inliers = ", sum(inliers)) # Ideally close to total_points



#Close Camera
ThorCamCSC.closecamera(test_cam) #returns 0 if succesful

#Delete Camera SDK
ThorCamCSC.thorcamsdkuninit() #returns 0 if succesful

#Close SLM SDK
Meadowlark.closesdk()








# ==============================================================================
#Helper Functions
# ==============================================================================


#First need to find how centers of SLM map to camera, user needs to manualy change (cx, cy)
#84/255 is white, 53/255 is black, generates a black circle. 
center_spot_mask = fill(84/255, 1024, 1024)
N = 1024
bg = 84/255
fg = 53/255
r = 12
cx = 512#514, now 514
cy = 512#592, now 561
#Note to self (512, 512)_SLM maps to (713, 429)_Camera. 
#and (514, 561)_SLM maps to (720, 540)_Camera aka the center of the camera. 
#so the scaling factor is $\frac{\sqrt{(720-713)^2 + (540-429)^2}}{\sqrt{(561-512)^2 + (514-512)^2}} \approx 2.27$, 
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
    
    # floor(1080/2.19/32) = 15, which is the maximum number of whole squares that can fit in the camera's vertical FOV
    # 15 * 32 gives a 480 pixel grid on the SLM, which maps to 480*2.19 = 1051 pixels on the camera, fitting within the 1080 pixel height.
    # 15x15 squares gives a 480x480 patch, resulting in 14x14 internal corners
    squares_per_side = 13  
    #need to vary this size and check error in affine transformation. 
    center_x = 514 # set these to the empirically determined center of the SLM, which maps to the center of the camera
    center_y = 561 
    
    # Define bounds centered precisely on (514, 560)
    x_start = center_x - (squares_per_side) * slm_square_size / 2 # 514 - (15*32)/2 = 274
    x_end   = x_start + squares_per_side * slm_square_size # 274 + 15*32 = 754
    y_start = center_y - (squares_per_side) * slm_square_size / 2 # 560 - (15*32)/2 = 320
    y_end   = y_start + squares_per_side * slm_square_size # 320 + 15*32 = 800

    # Initialize the matrix with your baseline dark voltage 
    phaseGrid = fill(v_dark, N, N)

    #= # This code is to generate the checkboard for corner tracking. No longer needed since used center of squares instead. 
    # Populate the checkerboard using column-major looping 
    for row in y_start:(y_end - 1)
        block_y = (row - y_start) ÷ slm_square_size
        
        for col in x_start:(x_end - 1)
            block_x = (col - x_start) ÷ slm_square_size
            
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
            pt_x = x_start + (c + 1) * slm_square_size
            pt_y = y_start + (r + 1) * slm_square_size

            slm_points[1, 1, n] = Float32(pt_x) # X Coordinate
            slm_points[2, 1, n] = Float32(pt_y) # Y Coordinate
            n += 1
        end
    end=#
    
    slm_pts_vec = Vector{Tuple{Float32, Float32}}()

    # To ensure a perfect match with the camera-side sorting trajectory:
    # Loop through rows (Y-axis) first, then columns (X-axis)
    for by in 0:(squares_per_side - 1)
        for bx in 0:(squares_per_side - 1)
            
            # Identify active white squares
            if (bx + by) % 2 == 1
                col_start = round(Int, x_start + bx * slm_square_size)
                row_start = round(Int, y_start + by * slm_square_size)
                
                # Project the square onto the SLM matrix
                phaseGrid[col_start:(col_start + slm_square_size - 1), row_start:(row_start + slm_square_size - 1)] .= v_bright
                
                # Compute the absolute center point of this specific white square
                pt_x = x_start + bx * slm_square_size + (slm_square_size / 2)
                pt_y = y_start + by * slm_square_size + (slm_square_size / 2)
                
                push!(slm_pts_vec, (Float32(pt_x), Float32(pt_y)))
            end
        end
    end

    # Format into an OpenCV-compatible 3D tensor: (2, 1, TotalPoints)
    total_whites = length(slm_pts_vec)
    slm_points = zeros(Float32, 2, 1, total_whites)
    for i in 1:total_whites
        slm_points[1, 1, i] = slm_pts_vec[i][1] # X Coordinate
        slm_points[2, 1, i] = slm_pts_vec[i][2] # Y Coordinate
    end

    return (phaseGrid, slm_points)
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


#If need to convert from 2d to 3d for OpenCV, here is how:
#diff_uint8_3d = reshape(diff_uint8', 1, size(diff_uint8', 1), size(diff_uint8', 2))