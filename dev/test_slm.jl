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

println("Successfully built slm_points tensor with shape: ", size(slm_points)) # Returns (2, 1, 112)

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

# succesfful below returns:               ThorCamCSCCamera(Ptr{Nothing}(0x0000015822f1d4c0), 40, 1000, UInt8[0x33, 0x37, 0x34, 0x33, 0x30], 5, CameraFormat(1440, 1080, 1.0, 0.0, "CMOS"), 2, 0, 1, 30, CameraROI(1, 1, 1440, 1080), 10, MicroscopeControl.HardwareImplementations.ThorCamCSC.SINGLE_FRAME, MicroscopeControl.HardwareImplementations.ThorCamCSC.SOFTWARE_TRIGGER, 0, 0, 0)
test_cam = ThorCamCSC.ThorCamCSCCamera()
test_cam.exposure_time = Clonglong(38897) # 38.897 ms

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



# SKIP UINT8 CONVERSION: Instead, scale strictly to [0.0, 1.0] Float64
norm_gray = diff_float ./ robust_max

slm_square_size = 32 
optical_scaling = 2.164 #this number was determined empircally with the helper functions
ideal_camera_width = slm_square_size * optical_scaling # ~70

# The box filter kernel size must be an odd integer to have a perfect center pixel
grid_size = round(Int, ideal_camera_width)
if grid_size % 2 == 0
    grid_size += 1 # Force it to be odd (e.g., 71 pixels)
end

# 2. Apply the normalized box filter
# Dividing by the area (grid_size^2) keeps the filtered intensities bounded between 0.0 and 1.0
h1 = ones(grid_size, grid_size)
img_filtered = imfilter(norm_gray, centered(h1))

# 3. Locate the sharp peak centers
localmax = mapwindow(maximum, img_filtered, (grid_size, grid_size)) .== img_filtered
img_peaks = localmax .* img_filtered

# Since the peaks are now sharp pyramids scaled to [0.0, 1.0], 
# a fractional threshold easily isolates the true centers.
threshold = 50 #Determined Empirically
pts = findall(img_peaks .> threshold)






# ==============================================================================
# FIXED CAMERA SORTING LOOP: Group by Row (Y = p[1]), sort by Column (X = p[2])
# This perfectly matches the row-major trajectory of your SLM generator
# ==============================================================================
y_tolerance = 15
unique_y = sort(unique([p[1] for p in pts])) # Group by Row index (Vertical position)
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
    # Filter points belonging to the current horizontal row stripe
    pts_in_group = filter(p -> p[1] in y_group, pts)
    # Sort left-to-right along the Column index (Horizontal position)
    pts_sorted = sort(pts_in_group, by=p -> p[2]) 
    
    for p in pts_sorted
        push!(x_coords, Float32(p[2])) # True Space X = Column index
        push!(y_coords, Float32(p[1])) # True Space Y = Row index
    end
end

# ==============================================================================
# FIXED VISUALIZATION: Force Makie to display standard Top-Left Image Space
# ==============================================================================
fig = Figure(size=(800, 600))
# yreversed=true puts 0 at the top, matching standard camera coordinate space
ax = CM.Axis(fig[1, 1], title="Grid Center Detection (True Top-Left Space)", yreversed=true)

# Transposing norm_gray ensures rows map to the Y axis and columns map to X
heatmap!(ax, norm_gray', colormap=:grays, colorrange=(0, 0.3))
scatter!(ax, x_coords, y_coords, color=:transparent, markersize=10, strokecolor=:red, strokewidth=2)
display(fig)

# Estimate grid size from horizontal spacing changes
x_diff = diff(x_coords)
grid_size_est = Statistics.mean(x_diff[x_diff .> 20])

# ==============================================================================
# OPENCV AFFINE MATRICES PIPELINE
# ==============================================================================
total_points = length(x_coords)
camera_points = zeros(Float32, 2, 1, total_points)

for i in 1:total_points
    camera_points[1, 1, i] = x_coords[i] 
    camera_points[2, 1, i] = y_coords[i] 
end

println("Successfully constructed camera matrix with $total_points points.")

if size(camera_points, 3) != size(slm_points, 3)
    error("Matrix size mismatch! Camera found $(size(camera_points,3)) peaks, but SLM expects $(size(slm_points,3)) centers.")
else
    # Explicitly cast method parameter to Int32 and dimensions to UInt64 to solve the CxxWrap bug
    affine_matrix, inliers = cv.estimateAffine2D(
        slm_points,
            ;
        method=Int64(cv.RANSAC),
        ransacReprojThreshold=3.0,
        maxIters=UInt64(2000),
        confidence=0.99,
        refineIters=UInt64(10)
    )
    println("Affine transformation matrix successfully calculated!")
end






























# Group by y with ±7.5 tolerance, then sort x within each y group
y_tolerance = 15
unique_y = sort(unique([p[2] for p in pts]))
y_groups = []
for y_val in unique_y
    if isempty(y_groups) || (y_val - y_groups[end][1] > y_tolerance)
        push!(y_groups, [y_val])
    else
        push!(y_groups[end], y_val)
    end
end

x_coords = []
y_coords = []
for y_group in y_groups
    pts_in_group = filter(p -> p[2] in y_group, pts)
    pts_sorted = sort(pts_in_group, by=p -> p[1])
    for p in pts_sorted
        push!(x_coords, p[1])
        push!(y_coords, p[2])
    end
end

# Visualize results
fig = Figure(size=(1080, 1440))
ax = CM.Axis(fig[1, 1], title="grid center detection")
heatmap!(ax, Float64.(norm_gray), colormap=:grays, colorrange=(0, 0.3))
scatter!(ax, x_coords, y_coords, color=:transparent, markersize=10,strokecolor=:red, strokewidth=2)
fig

# Estimate grid size from y-coordinates
y_diff = diff(y_coords)
grid_size_est = Statistics.mean(y_diff[y_diff.>20]')


# 3. Format into the 3D Tensor expected by OpenCV's estimateAffine2D
total_points = length(x_coords)
camera_points = zeros(Float32, 2, 1, total_points)

for i in 1:total_points
    camera_points[1, 1, i] = x_coords[i] # X row
    camera_points[2, 1, i] = y_coords[i] # Y row
end

println("Successfully constructed camera matrix with $total_points points.")


if size(camera_points, 3) != size(slm_points, 3)
    error("Matrix size mismatch! Camera found $(size(camera_points,3)) peaks, but SLM expects $(size(slm_points,3)) centers.")
else
    # Run the calibration solver
    #affine_matrix, inliers = cv.estimateAffine2D(slm_points, camera_points)
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














#A) this is og filitering 
max_intensity = maximum(diff_float)
if max_intensity > 0
    diff_uint8 = round.(UInt8, (diff_float ./ max_intensity) .* 255)
else
    error("Subtracted matrix is blank. Verify laser/beam path.")
end

#B) new filtering 
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





gray = Float64.(diff_uint8) ./ 255












#C) This is another way to filter. 
sorted_pixels = sort(vec(diff_float))
len = length(sorted_pixels)

p_dark   = sorted_pixels[round(Int, 0.05 * len) + 1] # 5th percentile
p_bright = sorted_pixels[round(Int, 0.95 * len)]     # 95th percentile

if p_bright <= p_dark
    error("Image contrast is too low. Verify that the laser is unblocked and hitting the camera sensor.")
end

# Clamp the matrix tightly to this high-contrast window
diff_clipped = clamp.(diff_float, p_dark, p_bright)

# Scale precisely to fill the entire 8-bit dynamic range [0, 255]
diff_uint8 = round.(UInt8, ((diff_clipped .- p_dark) ./ (p_bright - p_dark)) .* 255)



#D) 
sorted_pixels = sort(vec(diff_float))
robust_max = sorted_pixels[round(Int, 0.99 * length(sorted_pixels))]
diff_float[diff_float .> robust_max] .= robust_max
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

#diff_uint8_3d = cv.blur(diff_uint8_3d, cv.Size{Int32}(Int32(7), Int32(7)))
#diff_uint8_3d = cv.blur(diff_uint8_3d, cv.Size{Int32}(Int32(5), Int32(5)))
#diff_uint8_3d = cv.blur(diff_uint8_3d, cv.Size{Int32}(Int32(3), Int32(3)))
#diff_uint8_3d = cv.GaussianBlur(diff_uint8_3d, cv.Size{Int32}(Int32(5), Int32(5)), 0.0)
diff_uint8_3d = cv.medianBlur(diff_uint8_3d, 7)

# 5. MORPHOLOGICAL CLEANUP
# We create a small 3x3 structuring element to close up any remaining tiny gaps
kernel_elem = cv.getStructuringElement(cv.MORPH_RECT, cv.Size{Int32}(Int32(5), Int32(5)))
diff_uint8_3d = cv.morphologyEx(diff_uint8_3d, cv.MORPH_CLOSE, kernel_elem)
diff_uint8_3d = cv.morphologyEx(diff_uint8_3d, cv.MORPH_OPEN, kernel_elem)

# 6. SUB-PIXEL GRADIENT RESTORATION
# Now that the squares are perfectly solid blocks, we apply a gentle 5x5 
# Gaussian blur. This softens the sharp edges into clean, linear slopes 
# so the Fourier sector-detector can calculate precise sub-pixel intersections.
diff_uint8_3d = cv.GaussianBlur(diff_uint8_3d, cv.Size{Int32}(Int32(5), Int32(5)), 0.0)

imshow(Float16.(dropdims(diff_uint8_3d, dims=1))') # Transpose back for correct orientation in display



#just to see if findchessboard can work with an ideal chessboard
#it does (storta, off by 2 pixels for some reason, but it does find the corners)
#control: diff_uint8 = round.(UInt8, (phaseGrid').* 255)




# ==============================================================================
#6. Use OpenCV to determine cordinates of the grid corners in the camera image, 
#and use these coordinates to compute the affine transformation between the SLM and the camera. 
#This transformation can then be used to map any point on the SLM to its corresponding point in the camera image.
# ==============================================================================
pattern_size = cv.Size{Int32}(Int32(14), Int32(14))

success, camera_points = cv.findChessboardCornersSB(diff_uint8_3d, pattern_size)


############################################
#trying claude now
# ── STAGE 1: Aggressive speckle destruction ──────────────────────────────────
# Squares are ~70px in camera space. A 21px median will kill speckle
# while barely touching the square boundaries.
diff_uint8_3d = reshape(diff_uint8', 1, size(diff_uint8', 1), size(diff_uint8', 2))


# Follow with a box blur to average out the residual salt-and-pepper artifacts
#diff_uint8_3d = cv.blur(diff_uint8_3d, cv.Size{Int32}(Int32(5), Int32(5)))

# ── STAGE 2: Binarize using Otsu ─────────────────────────────────────────────
# Otsu automatically finds the optimal threshold between your two gray levels.
# This converts the noisy grayscale into a clean black/white board.
_, diff_uint8_3d = cv.threshold(
    diff_uint8_3d, 0.0, 255.0,
    cv.THRESH_BINARY + cv.THRESH_OTSU
)

diff_uint8_3d = cv.medianBlur(diff_uint8_3d, 3)
diff_uint8_3d = cv.adaptiveThreshold(
    diff_uint8_3d, 255.0,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    Int64(11),
    2.0
)

# ── STAGE 3: Morphological cleanup on the binary image ───────────────────────
# Larger kernel now that we're working on solid binary squares
kernel_elem = cv.getStructuringElement(
    cv.MORPH_RECT, cv.Size{Int32}(Int32(9), Int32(9))
)   
diff_uint8_3d = cv.morphologyEx(diff_uint8_3d, cv.MORPH_CLOSE, kernel_elem)
diff_uint8_3d = cv.morphologyEx(diff_uint8_3d, cv.MORPH_OPEN, kernel_elem)

# ── STAGE 4: Sub-pixel gradient restoration ───────────────────────────────────
# Soften the now-perfect hard edges so findChessboardCornersSB can lock on
# sub-pixel saddle points. Keep this small — 5 or 7 is fine.
diff_uint8_3d = cv.GaussianBlur(
    diff_uint8_3d, cv.Size{Int32}(Int32(5), Int32(5)), 0.0
)

diff_float = Float64.(dropdims(diff_uint8_3d, dims=1)) # Convert back to 2D for display and corner detection
sorted_pixels = sort(vec(diff_float))
robust_max = sorted_pixels[round(Int, 0.99 * length(sorted_pixels))]
diff_float[diff_float .> robust_max] .= robust_max
diff_uint8 = round.(UInt8, (diff_float ./ robust_max) .* 255)
imshow(diff_float') # Transpose for correct orientation in display

# --- STEP 6: Corner detection with exhaustive flags ---
pattern_size = cv.Size{Int32}(Int32(sqrt(size(slm_points, 3))), Int32(sqrt(size(slm_points, 3)))) 
pattern_size = cv.Size(Int32(7), Int32(7))

# Use ALL the helper flags — SB is much more robust with these enabled
flags = Int64(cv.CALIB_CB_NORMALIZE_IMAGE | 
               cv.CALIB_CB_EXHAUSTIVE)

success, camera_points = cv.findChessboardCornersSB(diff_uint8_3d, pattern_size, flags=flags)

imshow(Float16.(dropdims(diff_uint8_3d, dims=1)')) # Transpose for correct orientation in display












println(Array(camera_points))

if !success
    error("OpenCV failed to find corners. Check image exposure levels.")
end

println("Grid intersections successfully locked by OpenCV!")
affine_matrix, inliers = cv.estimateAffine2D(slm_points, camera_points)
    
println("\n>>> SUCCESS: 2x3 Geometric Affine Matrix Computed <<<")
display(affine_matrix)







#Close Camera
ThorCamCSC.closecamera(test_cam) #returns 0 if succesful

#Delete Camera SDK
ThorCamCSC.thorcamsdkuninit() #returns 0 if succesful

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
cy = 512#592
#Note to self (512, 512)_SLM maps to (713, 367)_Camera. 
#and (514, 592)_SLM maps to (720, 540)_Camera aka the center of the camera. 
#so the scaling factor is $\frac{\sqrt{(720-713)^2 + (540-367)^2}}{\sqrt{(592-512)^2 + (514-512)^2}} \approx 2.16$, 
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
    center_x = 514 # set these to the empirically determined center of the SLM, which maps to the center of the camera
    center_y = 592 
    
    # Define bounds centered precisely on (514, 560)
    x_start = center_x - (squares_per_side) * square_size / 2 # 514 - (15*32)/2 = 274
    x_end   = x_start + squares_per_side * square_size # 274 + 15*32 = 754
    y_start = center_y - (squares_per_side) * square_size / 2 # 560 - (15*32)/2 = 320
    y_end   = y_start + squares_per_side * square_size # 320 + 15*32 = 800

    # Initialize the matrix with your baseline dark voltage 
    phaseGrid = fill(v_dark, N, N)

    #=
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
    end=#
    
    slm_pts_vec = Vector{Tuple{Float32, Float32}}()

    # To ensure a perfect match with the camera-side sorting trajectory:
    # Loop through rows (Y-axis) first, then columns (X-axis)
    for by in 0:(squares_per_side - 1)
        for bx in 0:(squares_per_side - 1)
            
            # Identify active white squares
            if (bx + by) % 2 == 1
                col_start = round(Int, x_start + bx * square_size)
                row_start = round(Int, y_start + by * square_size)
                
                # Project the square onto the SLM matrix
                phaseGrid[col_start:(col_start + square_size - 1), row_start:(row_start + square_size - 1)] .= v_bright
                
                # Compute the absolute center point of this specific white square
                pt_x = x_start + bx * square_size + (square_size / 2)
                pt_y = y_start + by * square_size + (square_size / 2)
                
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

test_cam.exposure_time = Clonglong(40000.0) # microseconds, so this is 40 ms



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