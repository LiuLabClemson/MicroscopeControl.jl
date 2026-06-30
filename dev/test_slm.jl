using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.ThorCamCSC
using ImageView
using Images

using MicroscopeControl.HardwareImplementations.Meadowlark # for the SLM
using OpenCV
const cv = OpenCV

using CairoMakie
CM = CairoMakie
using Statistics
using JLD2
using Dates 
using LinearAlgebra
using Printf


#Constants
const slm_square_size = 32 
const optical_scaling = 2.19 #this number was determined empircally with the helper functions



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

slm.phase = phaseGrid # (x, y)
Meadowlark.writesingleimage(slm) # returns 1 for success
imshow(slm.phase')


# ==============================================================================
#3. Setup camera and trigger it to capture an image of the pattern displayed on the SLM. 
# In future, may save this image as "calib_frame_v$(v).png", where $(v) is the voltage value used for that frame.
# Always disarm the camera after each capture, otherwise the camera will still be waiting for the next trigger 
#and will not capture a new image.
# ==============================================================================

test_cam = ThorCamCSC.ThorCamCSCCamera()
#test_cam.exposure_time = Clonglong(38897) # 38.897 ms, emprically determined 

test_cam.exposure_time = Clonglong(150005) 



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
#Filter any peaks too close together
min_separation = grid_size / 2.0  # Safe distance threshold (~35 pixels)
cleaned_pts = CartesianIndex{2}[]

for p in pts
    # Check if this peak is too close to any peak we've already accepted
    is_duplicate = false
    for accepted_p in cleaned_pts
        dist = sqrt((p[1] - accepted_p[1])^2 + (p[2] - accepted_p[2])^2)
        if dist < min_separation
            is_duplicate = true
            break
        end
    end
    
    # If it's a unique square, keep it
    if !is_duplicate
        push!(cleaned_pts, p)
    end
end
pts = cleaned_pts  # Overwrite with the unique, cleaned points array



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
      color = :black, 
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




#-------------------------------------------------------------------
#The following code is just to visulize and diagonse affine matrix, do not normally run:
N = size(slm_points, 3)

pred_x = zeros(Float64, N)
pred_y = zeros(Float64, N)

for i in 1:N
    xs = slm_points[1,1,i]
    ys = slm_points[2,1,i]

    pred_x[i] =
        affine_matrix[1,1,1] * xs +
        affine_matrix[1,2,1] * ys +
        affine_matrix[1,3,1]

    pred_y[i] =
        affine_matrix[1,1,2] * xs +
        affine_matrix[1,2,2] * ys +
        affine_matrix[1,3,2]
end

obs_x = vec(camera_points[1,1,:])
obs_y = vec(camera_points[2,1,:])

slm_x = vec(slm_points[1,1,:])
slm_y = vec(slm_points[2,1,:])


#Visulization 1:
fig = Figure(size=(1400,600))

# -------------------------
# LEFT: SLM coordinate space
# -------------------------
ax1 = CM.Axis(
    fig[1,1],
    title = "SLM Coordinate Space",
    xlabel = "SLM X",
    ylabel = "SLM Y",
    yreversed = true
)

scatter!(
    ax1,
    slm_x,
    slm_y,
    color = :blue,
    markersize = 10
)

# Optional numbering
labels = string.(1:length(slm_x))

text!(
    ax1,
    slm_x,
    slm_y,
    text = labels,
    fontsize = 10,
    color = :black
)

# -------------------------
# RIGHT: Camera coordinate space
# -------------------------
ax2 = CM.Axis(
    fig[1,2],
    title = "Affine Fit: Predicted vs Observed Camera Centers",
    xlabel = "Camera X (pixels)",
    ylabel = "Camera Y (pixels)",
    yreversed = true
)

# observed centers
scatter!(
    ax2,
    obs_x,
    obs_y,
    color = :limegreen,
    markersize = 12,
    label = "Observed"
)

# predicted centers
scatter!(
    ax2,
    pred_x,
    pred_y,
    color = :red,
    marker = :x,
    markersize = 14,
    label = "Affine Prediction"
)

# residual vectors
for i in eachindex(obs_x)

    lines!(
        ax2,
        [pred_x[i], obs_x[i]],
        [pred_y[i], obs_y[i]],
        color = :orange,
        linewidth = 2
    )

end

axislegend(ax2)

display(fig)

#visulization 2:
scale_factor = 20
fig = Figure(size=(900,700))

ax = CM.Axis(
    fig[1,1],
    title = "Affine Residual Field (20× Magnified)",
    xlabel = "Camera X",
    ylabel = "Camera Y",
    yreversed = true
)

scatter!(
    ax,
    obs_x,
    obs_y,
    color = :black,
    markersize = 8
)

for i in eachindex(obs_x)

    dx = obs_x[i] - pred_x[i]
    dy = obs_y[i] - pred_y[i]

    lines!(
        ax,
        [obs_x[i],
         obs_x[i] + scale_factor*dx],

        [obs_y[i],
         obs_y[i] + scale_factor*dy],

        color = :red,
        linewidth = 2
    )
end

display(fig)

#visulization 3:
fig = Figure(size=(900,700))

ax = CM.Axis(
    fig[1,1],
    title = "Observed vs Affine-Predicted Grid Centers",
    yreversed = true
)

heatmap!(
    ax,
    norm_gray',
    colormap = :grays,
    colorrange = (0,0.3)
)

scatter!(
    ax,
    obs_x,
    obs_y,
    color = :lime,
    markersize = 12,
    label = "Observed"
)

scatter!(
    ax,
    pred_x,
    pred_y,
    color = :red,
    marker = :cross,
    markersize = 14,
    label = "Predicted"
)

for i in eachindex(obs_x)
    lines!(
        ax,
        [pred_x[i], obs_x[i]],
        [pred_y[i], obs_y[i]],
        color = :yellow
    )
end

axislegend(ax)

display(fig)
#-------------------------------------------------------------------







# ==============================================================================
#7. Sweep the SLm in 256 steps, and measure the corresponding camera intensity on a 2 by 2 block
# I have three ideas to do this. ethod (A) is to use bilinear interpolation with the 4 closest pixels. This would give the intensity at the transformed
# point as opposed to the integral of the intensity over the footprint; however, this should be a good first order approximation
# that should give a good smooth curve. Method (B), compute the camera coordiante that corresponds to the slm pixel, 
# then compute the average intensity of the 2 by 2 grid. The third method, (C), fits a gaussian distribution to 10 spots, and then
# uses this distribution to weight the camera intensities. This works the best but takes the longest!
# ==============================================================================

# Define the target calibration region on your SLM (e.g., a 400x400 central patch)
#Center of SLM that maps to center of Camera
target = [1440/2; 1080/2]
A = [
    affine_matrix[1,1,1]  affine_matrix[1,2,1];
    affine_matrix[1,1,2]  affine_matrix[1,2,2]
]
t = [affine_matrix[1,3,1]; affine_matrix[1,3,2]]
xy = A \ (target .- t)
x, y = round.(Int, A \ (target .- t))


ROISize = 475 #This is the dimension of the SLM that is being calibrated. 
# The size right now is aribtiary but obviously needs to fit in the camera
half = (ROISize - 1) ÷ 2
startx = max(1, x - half); endx = min(1024, x + half)
starty = max(1, y - half); endy = min(1024, y + half)
slm_x_range = startx:endx
slm_y_range = starty:endy

num_slm_x = length(slm_x_range)
num_slm_y = length(slm_y_range)


#set exposure time lower
#test_cam.exposure_time = Clonglong(20008) # 38.897 ms, emprically determined 
test_cam.exposure_time = Clonglong(70005)




# -----------------------------
# Method (A) 

intensity_cube = zeros(Float32, num_slm_x, num_slm_y, 256)

println("Initializing 256-Step Phase Calibration Sweep...")




for v in 0:255
    # Normalize voltage step to [0.0, 1.0] for the SDK
    v_norm = v / 255.0
    slm.phase = fill(v_norm, 1024, 1024)
    Meadowlark.writesingleimage(slm)
    
    # Allow the liquid crystals to mechanically settle completely (~30-50ms)
    #sleep(0.04) 
    
    # Capture the frame
    cam_frame = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end
    
    # Cast raw ADU image directly to Float32
    cam_img = Float32.(cam_frame)


    
    
    # Loop over every SLM pixel in the defined region
    for (xi, xs) in enumerate(slm_x_range)
        for (yi, ys) in enumerate(slm_y_range)
            
            # Map the SLM pixel center to floating-point Camera coordinates
            xc = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
            yc = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
            
            # Find the top-left pixel index of the surrounding 2x2 cluster
            x0 = floor(Int, xc)
            y0 = floor(Int, yc)
            x1 = x0 + 1
            y1 = y0 + 1
            
            # Bounds check to protect against edge overflows on a 1080x1440 sensor
            if (1 <= x0 && x1 <= 1440 && 1 <= y0 && y1 <= 1080)
                # Determine sub-pixel fractional weights
                fx = xc - x0
                fy = yc - y0
                
                # Fetch intensities of the 4 surrounding CMOS pixels
                i00 = cam_img[y0, x0]
                i10 = cam_img[y0, x1]
                i01 = cam_img[y1, x0]
                i11 = cam_img[y1, x1]
                
                # Compute the bilinear interpolation (Weighted Average)
                interp_intensity = (1 - fx) * (1 - fy) * i00 +
                                   fx * (1 - fy) * i10 +
                                   (1 - fx) * fy * i01 +
                                   fx * fy * i11
                                   
                intensity_cube[xi, yi, v + 1] = interp_intensity
            else
                # Fallback safeguard for out-of-bounds mapping
                intensity_cube[xi, yi, v + 1] = 0.0f0
            end
        end
    end
    
    if v % 32 == 0
        println("Sweep Progress: Voltage Step $v/255 Captured and Mapped.")
    end
end

println("Step 7 Complete! Data cube locked with shape: ", size(intensity_cube))






# -----------------------------
# Method (B) --- similar results to method (A)...
# Precompute Camera Coordinates for Every SLM Pixel
xc_map = zeros(Float32, num_slm_x, num_slm_y)
yc_map = zeros(Float32, num_slm_x, num_slm_y)

for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)

        xc_map[xi, yi] =
            affine_matrix[1,1,1] * xs +
            affine_matrix[1,2,1] * ys +
            affine_matrix[1,3,1]

        yc_map[xi, yi] =
            affine_matrix[1,1,2] * xs +
            affine_matrix[1,2,2] * ys +
            affine_matrix[1,3,2]

    end
end

# Pre-allocate the Intensity Data Cube using Float32 to optimize memory
intensity_cube = zeros(Float32, num_slm_x, num_slm_y, 256)
#Sweep
for v in 0:255

    v_norm = v / 255.0

    slm.phase .= v_norm
    Meadowlark.writesingleimage(slm)

    #Gives time for SLM to write image, note Blink_C_wrapper does have a write complete function but wasnt implemented in meadowlark_dev.jl
    #sleep(0.04)

    cam_frame = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end

    cam_img = Float32.(cam_frame)


    for xi in 1:num_slm_x
        for yi in 1:num_slm_y

            xc = xc_map[xi, yi]
            yc = yc_map[xi, yi]

            x0 = round(Int, xc)
            y0 = round(Int, yc)

            if 1 <= x0 < 1440 &&
               1 <= y0 < 1080

                intensity_cube[xi, yi, v+1] =
                    (
                        cam_img[y0,   x0]   +
                        cam_img[y0,   x0+1] +
                        cam_img[y0+1, x0]   +
                        cam_img[y0+1, x0+1]
                    ) * 0.25f0

            else
                intensity_cube[xi, yi, v+1] = 0f0
            end
        end
    end

    if v % 32 == 0
        println("Sweep Progress: Voltage Step $v/255 Captured and Mapped.")
    end
end





##USE THIS METHOD
#########$$$$$$$$$$METHOD C$$$$$$$$$$$$$$$$$$$$$$##############
# ==============================================================================
# STAGE 0: Narrow pre-sweep (steps 48–58) to locate the first minimum
# Purpose: find v_dark cheaply so we can build the dark stack before the
#          main sweep. Only 11 frames. Exposure + SLM write dominates cost.
# ==============================================================================
# In the presweep loop, replace the patch indexing with full ROI indexing:
presweep_voltages = collect(48:58)          # integer grey values, 11 frames
n_presweep        = length(presweep_voltages)
presweep_cube_full = zeros(Float32, num_slm_x, num_slm_y, n_presweep)

for (vi, v) in enumerate(presweep_voltages)
    slm.phase = fill(v / 255.0, 1024, 1024)
    Meadowlark.writesingleimage(slm)

    cam_frame = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end
    cam_img = Float32.(cam_frame)

    for (xi, xs) in enumerate(slm_x_range)
        for (yi, ys) in enumerate(slm_y_range)
            xc = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
            yc = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
            x0 = floor(Int, xc); y0 = floor(Int, yc)
            x1 = x0+1;           y1 = y0+1
            if 1 <= x0 && x1 <= 1440 && 1 <= y0 && y1 <= 1080
                fx = xc-x0; fy = yc-y0
                presweep_cube_full[xi, yi, vi] =
                    (1-fx)*(1-fy)*cam_img[y0,x0] + fx*(1-fy)*cam_img[y0,x1] +
                    (1-fx)*fy*cam_img[y1,x0]     + fx*fy*cam_img[y1,x1]
            end
        end
    end
end

# Per-pixel minimum voltage map
v_dark_map = zeros(Int, num_slm_x, num_slm_y)
for xi in 1:num_slm_x
    for yi in 1:num_slm_y
        min_vi = argmin(presweep_cube_full[xi, yi, :])
        v_dark_map[xi, yi] = presweep_voltages[min_vi]
    end
end

# ==============================================================================
# STAGE 1: Dark stack at v_dark
# ==============================================================================
 
 
phase_dark = fill(0.208316f0, 1024, 1024)

for xi in 1:num_slm_x
    for yi in 1:num_slm_y
        global_x = slm_x_range[xi]
        global_y = slm_y_range[yi]

        phase_dark[global_x, global_y] = v_dark_map[xi, yi] / 255
    end
end
slm.phase = phase_dark
Meadowlark.writesingleimage(slm)
 
n_dark_frames = 100
dark_accumulator = zeros(Float64, 1080, 1440)
 
for k in 1:n_dark_frames
    img = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end
    dark_accumulator .+= Float64.(img)
end
 
dark_mean = dark_accumulator ./ n_dark_frames
 
@printf("  Dark stack complete. Mean dark level = %.1f ADU\n", mean(dark_mean))
 
 
# ==============================================================================
# STAGE 2: Diffraction test — illuminate 10 best_idx SLM pixels, measure PSF
# ==============================================================================
 
println("\n=== STAGE 2: Diffraction test — PSF characterisation ===")
best_idx = sortperm(errors)[1:10]
 

# Build test phase: dark everywhere, bright only at the 10 calibration points
phase_diffraction = copy(phase_dark)
for idx in best_idx
    xs = round(Int, slm_points[1,1,idx])
    ys = round(Int, slm_points[2,1,idx])
    phase_diffraction[xs, ys] = 84/255
end
 
slm.phase = phase_diffraction
Meadowlark.writesingleimage(slm)
 
n_test_frames   = 100
test_accumulator = zeros(Float64, 1080, 1440)
 
for k in 1:n_test_frames
    img = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end
    test_accumulator .+= Float64.(img)
end
 
test_mean     = test_accumulator ./ n_test_frames
spot_img      = max.(test_mean .- dark_mean, 0.0)

imshow(test_mean)
imshow(spot_img)
#Note to self, in future might get better results if write/capture one dark then one test 100 times 

@printf("  Spot image: max=%.1f ADU, mean=%.2f ADU\n", maximum(spot_img), mean(spot_img))
 
# ── Predict camera centres from affine ──────────────────────────────────────
predicted_centers = Tuple{Float64,Float64}[]   # (xc=col, yc=row)

for idx in best_idx
    xs = slm_points[1,1,idx]
    ys = slm_points[2,1,idx]
    xc = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
    yc = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    push!(predicted_centers, (xc, yc))
end
 
# ── Extract centroid-refined ROIs ─────────────────────────────────────────────
window_big  = 8    # initial extraction window for centroid finding
window_fit  = 6    # final tight window for Gaussian fitting (±6 px = 13×13)
 
rois_fit         = Matrix{Float64}[]
true_centers_fit = Tuple{Float64,Float64}[]   # (col, row)
 
for (xc_pred, yc_pred) in predicted_centers
    pc = round(Int, xc_pred)
    pr = round(Int, yc_pred)
 
    r0 = clamp(pr - window_big, 1, 1080)
    r1 = clamp(pr + window_big, 1, 1080)
    c0 = clamp(pc - window_big, 1, 1440)
    c1 = clamp(pc + window_big, 1, 1440)
 
    big_roi = spot_img[r0:r1, c0:c1]
    total   = sum(max.(big_roi, 0.0))
 
    if total < 1.0
        push!(true_centers_fit, (Float64(pc), Float64(pr)))
        push!(rois_fit, spot_img[clamp(pr-window_fit,1,1080):clamp(pr+window_fit,1,1080),
                                  clamp(pc-window_fit,1,1440):clamp(pc+window_fit,1,1440)])
        continue
    end
 
    # Intensity-weighted centroid in full-image coordinates
    wrow = sum((r0:r1) .* vec(sum(max.(big_roi, 0.0), dims=2))) / total
    wcol = sum((c0:c1) .* vec(sum(max.(big_roi, 0.0), dims=1))) / total
 
    push!(true_centers_fit, (wcol, wrow))
 
    tc_r = round(Int, wrow)
    tc_c = round(Int, wcol)
 
    rs = clamp(tc_r - window_fit, 1, 1080)
    re = clamp(tc_r + window_fit, 1, 1080)
    cs = clamp(tc_c - window_fit, 1, 1440)
    ce = clamp(tc_c + window_fit, 1, 1440)
    push!(rois_fit, spot_img[rs:re, cs:ce])
end
 
# ── 2D Gaussian fit ──────────────────────────────────────────────────────────
"""
    fit_2d_gaussian_full(roi)
 
Fit a 2D Gaussian with 6 parameters:
    I(r,c) = floor + amplitude × exp(-½[(c-μ_c)²/σ_c² + (r-μ_r)²/σ_r²])
 
Parameters returned:
  μ_col, μ_row   : centre in ROI-local 1-based pixel coordinates
  σ_col, σ_row   : 1/e² half-widths (pixels)
  amplitude      : peak above floor (ADU)
  floor          : constant background level (ADU)
 
Uses intensity-weighted moments (robust, fast, no iterative solve needed).
"""
function fit_2d_gaussian_full(roi::Matrix{<:Real})
    nr, nc = size(roi)
 
    floor_val = minimum(roi)
    roi_bg    = max.(roi .- floor_val, 0.0)
    total     = sum(roi_bg)
 
    if total < 1e-9
        return (μ_col=nc/2.0, μ_row=nr/2.0,
                σ_col=NaN, σ_row=NaN,
                amplitude=NaN, floor=floor_val)
    end
 
    # First moments (centroid)
    μ_row = sum((1:nr) .* vec(sum(roi_bg, dims=2))) / total
    μ_col = sum((1:nc) .* vec(sum(roi_bg, dims=1))) / total
 
    # Second moments (variance)
    σ²_row = 0.0; σ²_col = 0.0
    for r in 1:nr, c in 1:nc
        w = roi_bg[r, c]
        σ²_row += w * (r - μ_row)^2
        σ²_col += w * (c - μ_col)^2
    end
    σ²_row /= total; σ²_col /= total
 
    amplitude = maximum(roi_bg)
 
    return (μ_col     = μ_col,
            μ_row     = μ_row,
            σ_col     = sqrt(σ²_col),
            σ_row     = sqrt(σ²_row),
            amplitude = amplitude,
            floor     = floor_val)
end
 
# Fit all 10 spots
gaussian_fits = [fit_2d_gaussian_full(roi) for roi in rois_fit]
 
println("\n  6-parameter Gaussian fits:")
println("  Spot | μ_col  | μ_row  | σ_col | σ_row | FWHM_c | FWHM_r | Amplitude |  Floor")
for (i, g) in enumerate(gaussian_fits)
    if isnan(g.σ_col)
        println("  $i    | FAILED")
        continue
    end
    @printf("  %4d | %6.2f | %6.2f | %5.2f | %5.2f | %6.2f  | %6.2f  | %9.1f | %6.1f\n",
            i, g.μ_col, g.μ_row, g.σ_col, g.σ_row,
            2.355*g.σ_col, 2.355*g.σ_row, g.amplitude, g.floor)
end
 
σ_cols_all = filter(!isnan, [g.σ_col for g in gaussian_fits])
σ_rows_all = filter(!isnan, [g.σ_row for g in gaussian_fits])
σ_col_final = median(σ_cols_all)
σ_row_final = median(σ_rows_all)
 
println("\n  Summary:")
@printf("  Median σ_col = %.2f px  (FWHM = %.2f px = %.2f µm)\n",
        σ_col_final, 2.355*σ_col_final, 2.355*σ_col_final*3.45)
@printf("  Median σ_row = %.2f px  (FWHM = %.2f px = %.2f µm)\n",
        σ_row_final, 2.355*σ_row_final, 2.355*σ_row_final*3.45)
 
# ── Visualisation: 10 spots with Gaussian overlay ────────────────────────────
fig_spots = Figure(size=(1400, 600))
 
for (i, (roi, g)) in enumerate(zip(rois_fit, gaussian_fits))
    row_panel = ceil(Int, i / 5)
    col_panel = mod1(i, 5)
    nr, nc    = size(roi)
 
    col_ax = (1:nc) .- g.μ_col    # zero-centred axes for display
    row_ax = (1:nr) .- g.μ_row
 
    ax = CM.Axis(fig_spots[row_panel, col_panel],
                  title   = "Spot $i",
                  xlabel  = "Δcol (px)",
                  ylabel  = "Δrow (px)",
                  aspect  = DataAspect())
 
    heatmap!(ax, col_ax, row_ax, roi', colormap=:inferno)
 
    if !isnan(g.σ_col)
        # Overlay Gaussian contours at 1σ, 2σ, 3σ
        θ = range(0, 2π, length=200)
        for nσ in [1.0, 2.0]
            lines!(ax,
                   nσ * g.σ_col .* cos.(θ),
                   nσ * g.σ_row .* sin.(θ),
                   color     = nσ == 1.0 ? :white : :cyan,
                   linewidth = 1.5,
                   linestyle = :dash)
        end
 
        # Annotate with 4 key parameters
        text!(ax,
              minimum(col_ax) + 0.5,
              minimum(row_ax) + 0.5,
              text     = @sprintf("σ=(%.1f,%.1f)\nA=%.0f\nf=%.0f",
                                   g.σ_col, g.σ_row, g.amplitude, g.floor),
              fontsize  = 9,
              color     = :white,
              align     = (:left, :bottom))
    end
end
 
display(fig_spots)
 
# ── Full image overlay: predicted (cyan) vs centroid (yellow) ────────────────
fig_overlay = Figure(size=(900, 700))
ax_ov = CM.Axis(fig_overlay[1,1],
                 title     = "Diffraction test overlay",
                 yreversed = true)
heatmap!(ax_ov, spot_img', colormap=:inferno)
scatter!(ax_ov,
         [p[1] for p in predicted_centers],
         [p[2] for p in predicted_centers],
         color=:cyan, markersize=14, label="affine prediction")
scatter!(ax_ov,
         [p[1] for p in true_centers_fit],
         [p[2] for p in true_centers_fit],
         color=:yellow, marker=:cross, markersize=12,
         strokewidth=2, label="centroid refined")
axislegend(ax_ov)
display(fig_overlay)
 
 
# ==============================================================================
# STAGE 3: Full 256-step Gaussian-weighted sweep — Method C
# ==============================================================================
 
println("\n=== STAGE 3: Full 256-step Method C sweep ===")
 
kh_col = ceil(Int, 3 * σ_col_final)
kh_row = ceil(Int, 3 * σ_row_final)
@printf("  Kernel: σ_col=%.2f, σ_row=%.2f → footprint %d×%d px\n",
        σ_col_final, σ_row_final, 2*kh_row+1, 2*kh_col+1)
 
# Pre-compute floating-point camera coordinate map (once, outside loop)
xc_map_C = zeros(Float64, num_slm_x, num_slm_y)
yc_map_C = zeros(Float64, num_slm_x, num_slm_y)
for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)
        xc_map_C[xi, yi] =
            affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc_map_C[xi, yi] =
            affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    end
end
 
"""
    gaussian_weighted_sample(cam_img, xc, yc, σ_col, σ_row, half_col, half_row)
 
Matched-filter intensity estimate at sub-pixel position (xc=col, yc=row).
Weights each camera pixel by the Gaussian PSF profile at its distance from
the true centre. Equivalent to a weighted mean with weights:
 
    w(r,c) = exp(-½ [(c - xc)²/σ_col² + (r - yc)²/σ_row²])
 
This is the maximum-likelihood intensity estimator for a Gaussian PSF
in the presence of Poisson/Gaussian shot noise.
"""
function gaussian_weighted_sample(cam_img::Matrix{Float32},
                                    xc::Float64, yc::Float64,
                                    σ_col::Float64, σ_row::Float64,
                                    half_col::Int, half_row::Int)
 
    cam_rows, cam_cols = size(cam_img)
 
    # Integer centre
    col0 = round(Int, xc)
    row0 = round(Int, yc)
 
    # Sub-pixel offset
    δcol = xc - col0
    δrow = yc - row0
 
    weighted_sum = 0.0f0
    weight_total = 0.0f0
 
    for dr in -half_row:half_row
        r = row0 + dr
        (r < 1 || r > cam_rows) && continue
 
        for dc in -half_col:half_col
            c = col0 + dc
            (c < 1 || c > cam_cols) && continue
 
            # Distance from the true (sub-pixel) centre
            d_col = (dc - δcol) / σ_col
            d_row = (dr - δrow) / σ_row
 
            w = exp((-0.5f0) * Float32(d_col^2 + d_row^2))
 
            weighted_sum += w * cam_img[r, c]
            weight_total += w
        end
    end
 
    return weight_total > (1.0f-9) ? weighted_sum / weight_total : (0.0f0)
end
 
intensity_cube = zeros(Float32, num_slm_x, num_slm_y, 256)
 
t_sweep_start = time()
 
for v in 0:255
    slm.phase = fill(v / 255.0, 1024, 1024)
    Meadowlark.writesingleimage(slm)
 
    cam_frame = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end
    cam_img = Float32.(cam_frame)
 
    for xi in 1:num_slm_x
        for yi in 1:num_slm_y
            intensity_cube[xi, yi, v+1] = gaussian_weighted_sample(
                cam_img,
                xc_map_C[xi, yi],
                yc_map_C[xi, yi],
                σ_col_final, σ_row_final,
                kh_col, kh_row)
        end
    end
 
    if v % 32 == 0
        elapsed = round(time() - t_sweep_start, digits=1)
        @printf("  Sweep: step %3d/255  |  elapsed %.1f s  |  est. remaining %.1f s\n",
                v, elapsed, elapsed / max(v,1) * (255 - v))
    end
end
 
total_time = round(time() - t_sweep_start, digits=1)
println("\nMethod C sweep complete in $(total_time) s.")
println("intensity_cube shape: ", size(intensity_cube))
println("Ready for Step 8 (extrema detection).")
 
# Quick sanity plot — centre pixel curve
mid_xi = num_slm_x ÷ 2
mid_yi = num_slm_y ÷ 2
curve_demo = Float64.(intensity_cube[mid_xi, mid_yi, :])
 
fig_demo = Figure(size=(800, 400))
ax_demo  = CM.Axis(fig_demo[1,1],
    title  = "Method C — centre pixel ($(slm_x_range[mid_xi]), $(slm_y_range[mid_yi]))",
    xlabel = "Voltage step",
    ylabel = "Gaussian-weighted intensity (ADU)")
lines!(ax_demo, 0:255, curve_demo, color=:crimson, linewidth=2)
display(fig_demo)
#########$$$$$$$$$$METHOD C_CLAUDE $$$$$$$$$$$$$$$$$$$$$$##############

















#Graph single plot
# Select an SLM pixel right in the middle of  calibrated Region of Interest
mid_xi = round(Int, num_slm_x / 2)
mid_yi = round(Int, num_slm_y / 2)


# Extract the 256-step voltage response curve for this specific pixel
raw_curve = intensity_cube[mid_xi, mid_yi, :]

# Retrieve its actual physical SLM coordinates for the plot title
actual_slm_x = slm_x_range[mid_xi]
actual_slm_y = slm_y_range[mid_yi]

# Generate the diagnostic figure
fig_diag = Figure(size=(800, 450))
ax_diag = CM.Axis(
    fig_diag[1, 1], 
    title="Raw Intensity Profile for SLM Pixel ($actual_slm_x, $actual_slm_y)",
    xlabel="Voltage Step (0 - 255)",
    ylabel="Camera Intensity (ADU)"
)

# Plot the continuous line and discrete data points
lines!(ax_diag, 0:255, raw_curve, color=:blue, linewidth=2, label="Bilinear Response")
scatter!(ax_diag, 0:255, raw_curve, color=:red, markersize=4)

# Display the plot window
display(fig_diag)

println("Diagnostic plot rendered for SLM Pixel ($actual_slm_x, $actual_slm_y).")
println("Verify that you see a clear oscillating wave with 3 maxima and 2 minima.")



 

#Graph several plots
# Pick a handful of pixel indices spread across your ROI
n_pixels_to_plot = 10
rng_xi = rand(1:num_slm_x, n_pixels_to_plot)
rng_yi = rand(1:num_slm_y, n_pixels_to_plot)

fig = Figure(size=(900, 500))
ax = CM.Axis(
    fig[1, 1],
    title = "Voltage Response — Multiple SLM Pixels Overlaid",
    xlabel = "Voltage Step (0–255)",
    ylabel = "Camera Intensity (ADU)",
    xticks = 0:10:255
)

colors = CM.cgrad(:turbo, n_pixels_to_plot, categorical=true)

for (i, (xi, yi)) in enumerate(zip(rng_xi, rng_yi))
    curve = intensity_cube[xi, yi, :]
    # Normalize each curve to its own max so shapes are comparable
    # even if absolute brightness differs pixel-to-pixel (comment out
    # this line if you want to compare raw, un-normalized intensities)
    curve_norm = curve ./ maximum(curve)

    actual_x = slm_x_range[xi]
    actual_y = slm_y_range[yi]

    lines!(ax, 0:255, curve_norm,
           color = colors[i],
           label = "($actual_x, $actual_y)")
end

axislegend(ax, position = :rb, framevisible = true)
display(fig)






















# ==============================================================================
#8. Fit parabola to 5 extremea 
# Fig 2.a
# ==============================================================================

# Custom lightweight moving average to smooth out camera noise per pixel
function smooth_profile(vector::Vector{Float32}, window::Int)
    smoothed = copy(vector)
    len = length(vector)
    for i in 1:len
        start_idx = max(1, i - window)
        end_idx = min(len, i + window)
        smoothed[i] = mean(@view vector[start_idx:end_idx])
    end
    return smoothed
end

# Define dimensions from your Step 7 intensity_cube
Nx, Ny, N_voltages = size(intensity_cube)

# Define target search window based on your clean data zone
# Voltage steps 40 to 115 map to Julia indices 41 to 116
# Empricallly determined for where the voltage response is most sinusoidial
search_indices = 41:116

# Storage matrices for the 5 extrema positions (now storing Float64 sub-pixel locations)
min1_map = zeros(Float64, Nx, Ny)
max1_map = zeros(Float64, Nx, Ny)
min2_map = zeros(Float64, Nx, Ny)
max2_map = zeros(Float64, Nx, Ny)
min3_map = zeros(Float64, Nx, Ny)
 
# Quality control tracking map: 4 = Full 4π, 2 = 2π fallback, 0 = Bad/Noisy
pixel_quality_map = zeros(Int, Nx, Ny)

# Analytical 3-point parabolic interpolation helper
function fit_parabolic_vertex(idx::Int, profile::AbstractVector)
    # Extract the three points around the integer extreme
    y_minus = profile[idx-1]
    y_zero  = profile[idx]
    y_plus  = profile[idx+1]
    
    denom = 2 * (y_minus + y_plus - 2 * y_zero)
    if denom == 0
        return Float64(idx) # Fallback to integer index if profile is flat
    end
    
    # Calculate sub-pixel offset from the center integer index
    offset = (y_minus - y_plus) / denom
    return idx + offset
end

println("Beginning Step 8: Topological Extrema Extraction with Parabolic Fitting...")

for xi in 1:Nx
    for yi in 1:Ny
        # 1. Extract and smooth the 1D voltage profile for this specific pixel
        raw_profile = intensity_cube[xi, yi, :]
        smoothed = smooth_profile(raw_profile, 2) # 5-point total smoothing window
        
        # 2. Extract all local minima and maxima strictly within our search window
        local_mins = Int[]
        local_maxes = Int[]
        
        for idx in (first(search_indices) + 1):(last(search_indices) - 1)
            # Check for local minimum
            if smoothed[idx] < smoothed[idx-1] && smoothed[idx] <= smoothed[idx+1]
                push!(local_mins, idx)
            # Check for local maximum
            elseif smoothed[idx] > smoothed[idx-1] && smoothed[idx] >= smoothed[idx+1]
                push!(local_maxes, idx)
            end
        end
        
        # 3. Filter out accidental double-peaks/noise jitters by enforcing sequential sorting
        try
            # Find Min1 (First clear dip around step 50)
            m1_idx = local_mins[findfirst(m -> m >= 45, local_mins)]
            
            # Find Max1 (First peak after Min1)
            idx_max1 = findfirst(mx -> mx > m1_idx, local_maxes)
            isnothing(idx_max1) && error()
            max1_idx = local_maxes[idx_max1]
            
            # Find Min2 (Second dip after Max1)
            idx_min2 = findfirst(m -> m > max1_idx, local_mins)
            isnothing(idx_min2) && error()
            m2_idx = local_mins[idx_min2]
            
            # Attempt to find Max2 and Min3 for full 4π calibration
            idx_max2 = findfirst(mx -> mx > m2_idx, local_maxes)
            idx_min3 = findfirst(m -> m > m2_idx + 10, local_mins) # Enforce distance from Min2
            
            # Note: We fit the parabola to the 'smoothed' profile to avoid latching onto high-frequency noise spikes
            if !isnothing(idx_max2) && !isnothing(idx_min3)
                max2_idx = local_maxes[idx_max2]
                m3_idx = local_mins[idx_min3]
                
                # Double check that sequence order is strictly preserved
                if m1_idx < max1_idx && max1_idx < m2_idx && m2_idx < max2_idx && max2_idx < m3_idx
                    min1_map[xi, yi] = fit_parabolic_vertex(m1_idx, smoothed)
                    max1_map[xi, yi] = fit_parabolic_vertex(max1_idx, smoothed)
                    min2_map[xi, yi] = fit_parabolic_vertex(m2_idx, smoothed)
                    max2_map[xi, yi] = fit_parabolic_vertex(max2_idx, smoothed)
                    min3_map[xi, yi] = fit_parabolic_vertex(m3_idx, smoothed)
                    pixel_quality_map[xi, yi] = 4 # Golden 4π pixel
                else
                    error() # Force fallback logic if sequence layout fails
                end
            else
                # Fallback to 2π Phase Modulation Mode (Only 2 segments found)
                min1_map[xi, yi] = fit_parabolic_vertex(m1_idx, smoothed)
                max1_map[xi, yi] = fit_parabolic_vertex(max1_idx, smoothed)
                min2_map[xi, yi] = fit_parabolic_vertex(m2_idx, smoothed)
                pixel_quality_map[xi, yi] = 2 # 2π Phase Stroke Mode
            end
            
        catch
            # Handle anomalous pixels that are completely unresolvable
            pixel_quality_map[xi, yi] = 0
        end
    end
end






# Display diagnostic summary
total_pixels = Nx * Ny
p4_count = count(q -> q == 4, pixel_quality_map)
p2_count = count(q -> q == 2, pixel_quality_map)
failed_count = count(q -> q == 0, pixel_quality_map)

println("--- Extrema Detection Summary ---")
println("Total Spatial Pixels Processed: $total_pixels")
println("Successfully Mapped 4π Pixels : $p4_count ($(round(p4_count/total_pixels*100, digits=2))%)")
println("Fallback 2π Modulated Pixels  : $p2_count ($(round(p2_count/total_pixels*100, digits=2))%)")
println("Failed / Dead Pixels Flagged  : $failed_count ($(round(failed_count/total_pixels*100, digits=2))%)")



#Plot to see 3 random slm pixles with max and min now labled correctly. 
n_pixels_to_plot = 3
rng_xi = rand(1:num_slm_x, n_pixels_to_plot)
rng_yi = rand(1:num_slm_y, n_pixels_to_plot)

fig = Figure(size=(1000, 600))
ax = CM.Axis(
    fig[1,1],
    title = "Voltage Response with Detected Extrema",
    xlabel = "Voltage Step",
    ylabel = "Normalized Intensity",
    xticks = 0:10:255
)

colors = CM.cgrad(:turbo, n_pixels_to_plot, categorical=true)



for (i, (xi, yi)) in enumerate(zip(rng_xi, rng_yi))

    curve = intensity_cube[xi, yi, :]
    curve_norm = curve ./ maximum(curve)  

    actual_x = slm_x_range[xi]
    actual_y = slm_y_range[yi]

    # Plot continuous line
    lines!(
        ax,
        0:255,
        curve_norm,
        color = colors[i],
        label = "($actual_x,$actual_y)"
    )

    # Gather precise Float64 extrema indices
    extrema_idx = [
        min1_map[xi, yi],
        max1_map[xi, yi],
        min2_map[xi, yi],
        max2_map[xi, yi],
        min3_map[xi, yi]
    ]

    # Remove zeros from failed detections
    extrema_idx = filter(x -> x > 0, extrema_idx)

    # FIX: Convert Float64 indices to Int ONLY for array indexing
    lookup_idx = round.(Int, extrema_idx)

    # Plot extrema markers at precise floating positions on X, but look up values via Int
    scatter!(
        ax,
        extrema_idx .- 1,          # exact float X-coordinate (voltage step)
        curve_norm[lookup_idx],    # fixed Y-coordinate lookup
        color = colors[i],
        markersize = 12
    )

    # Label extrema with coordinates
    text!(
        ax,
        extrema_idx .- 1,
        curve_norm[lookup_idx],
        text = fill("($actual_x,$actual_y)", length(extrema_idx)),
        fontsize = 10,
        align = (:left, :bottom),
        color = colors[i]
    )
end



axislegend(ax, position = :rb)

display(fig)


#SKIP THIS NOT NECCESSEARY ANY MORE WILL DELETE LATER
#---------------------------------------------------------------------------
#More diagonitic figures, dont normallly run this time for diffraction limit
#n
# This is for the diffraction test, not working yet. 
best_idx = sortperm(errors)[1:10]

nframes = 100

phase_dark = fill(0.208316, 1024, 1024)

for xi in 1:num_slm_x
    for yi in 1:num_slm_y

        if pixel_quality_map[xi,yi] > 0


            global_x = slm_x_range[xi]
            global_y = slm_y_range[yi]

            #(x, y)
            phase_dark[global_x, global_y] =
                (min1_map[xi,yi]-1)/255

        end

    end
end

phase_test = copy(phase_dark)

#phase_test = fill(53/255, 1024, 1024)

for idx in best_idx

    xs = round(Int, slm_points[1,1,idx])
    ys = round(Int, slm_points[2,1,idx])

    phase_test[
        xs,
        ys
    ] = 84/255

end


#capture average test image
slm.phase = phase_test
Meadowlark.writesingleimage(slm)

test_stack = Matrix{Float64}[]

for k in 1:nframes
    img = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end

    push!(test_stack, Float64.(img))
end

test_mean = zeros(size(test_stack[1]))

for img in test_stack
    test_mean .+= img
end

test_mean ./= nframes

#capture average dark image
slm.phase = phase_dark
Meadowlark.writesingleimage(slm)

dark_stack = Matrix{Float64}[]

for k in 1:nframes
    img = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end

    push!(dark_stack, Float64.(img))
end

dark_mean = zeros(size(dark_stack[1]))

for img in dark_stack
    dark_mean .+= img
end

dark_mean ./= nframes

#subtract the two
spot_img_rem = test_mean .- dark_mean

spot_img_rem[spot_img_rem .< 0] .= 0

imshow(spot_img_rem')
imshow(test_mean')

#=
mean1 = mean(dark_stack[1:50])
mean2 = mean(dark_stack[50:100])

diff = mean2 .- mean1

dark_stack_all = zeros(1080,1440,100)
for i in eachindex(dark_stack)
    dark_stack_all[:,:,i]=dark_stack[i]
end
imshow(dark_stack_all)
=#


# Fix dark stack movie visualization to match correct horizontal orientation

# --- Predict Spot Centers ---
predicted_centers = []
for idx in best_idx
    xs = slm_points[1,1,idx]
    ys = slm_points[2,1,idx]

    xc = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
    yc = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]

    push!(predicted_centers, (xc, yc))
end

rois = Matrix{Float64}[]


###OLD METHOD
window = 5

# --- ROI Extraction (CRITICAL FIX HERE) ---
for (xc, yc) in predicted_centers
    # TWEAK HERE: If your calibration matrix was generated using the old, un-transposed images, 
    # xc and yc are swapped relative to the true image matrix rows/columns.
    # If the spots are still missing, change this to: xi = round(Int, yc); yi = round(Int, xc)
    xi = round(Int, xc) # Column coordinate (Horizontal axis)
    yi = round(Int, yc) # Row coordinate (Vertical axis)

    # Ensure boundaries don't exceed image dimensions
    r_start = max(1, yi - window)
    r_end   = min(size(spot_img_rem, 1), yi + window)
    c_start = max(1, xi - window)
    c_end   = min(size(spot_img_rem, 2), xi + window)

    roi = spot_img_rem[r_start:r_end, c_start:c_end]

    local_bg = median(roi)
    roi_clean = roi .- local_bg
    roi_clean[roi_clean .< 0] .= 0

    push!(rois, roi_clean) # Pushing clean ROI so background subtraction is applied
end

# --- Visualization 1: Individual Spots ---
fig = Figure(size=(1200, 600))
for i in 1:length(rois)
    row = ceil(Int, i / 5)
    col = mod1(i, 5)

    ax = CM.Axis(fig[row, col], title="Spot $i")
    # Transpose the ROI so Makie displays it with correct x/y spatial orientation
    heatmap!(ax, rois[i]') 
end
display(fig)


# --- Compute FWHM on the first valid spot ---
peak = maximum(rois[1])

threshold = 0.05*peak

hit_pixels = count(rois[1] .> threshold)

profile_x = rois[1][(6+1),:]

# Find points above half maximum
halfmax = maximum(profile_x) / 2

inds = findall(profile_x .> halfmax)

fwhm_x = maximum(inds) - minimum(inds)

# Similarly for y
profile_y = rois[1][:,(6+1)]

halfmax = maximum(profile_y) / 2

inds = findall(profile_y .> halfmax)

fwhm_y = maximum(inds) - minimum(inds)


println("FWHM x = $fwhm_x px")
println("FWHM y = $fwhm_y px")


# --- Visualization 4: Full Overlaid Heatmap (FIXED SCATTER) ---
fig3 = Figure(size=(900, 700))
ax3 = CM.Axis(fig3[1, 1], yreversed=true, title="Diffraction Test Verification")

# Heatmap treats dim 1 as X, dim 2 as Y. Since spot_img_rem is [Row, Col], transposing is correct.
heatmap!(ax3, spot_img_rem') 

# FIX: Extract coordinates for ALL spots to plot simultaneously, rather than indexing [2]
all_xc = [p[1] for p in predicted_centers]
all_yc = [p[2] for p in predicted_centers]

scatter!(
    ax3,
    all_xc,
    all_yc,
    color=:cyan,
    markersize=15,
    strokecolor=:black,
    strokewidth=1
)
display(fig3)


peak_idx = argmax(spot_img_rem)
println("True peak row (y) and col (x): ", peak_idx)


#NEW METHOd
#claude way to visulize diffraction test
# ── Improved ROI extraction and centering ─────────────────────────────────

window = 8   # extract a larger initial window for centroid finding

rois_centered = Matrix{Float64}[]
true_centers  = Tuple{Float64,Float64}[]   # (col, row) in full image coords
predicted_centers_rc = Tuple{Int,Int}[]    # (row, col) from affine

for (xc, yc) in predicted_centers
    # xc = predicted column, yc = predicted row (from your affine convention)
    pred_col = round(Int, xc)
    pred_row = round(Int, yc)

    push!(predicted_centers_rc, (pred_row, pred_col))

    # Extract generous window around predicted center
    r0 = clamp(pred_row - window, 1, size(spot_img_rem, 1))
    r1 = clamp(pred_row + window, 1, size(spot_img_rem, 1))
    c0 = clamp(pred_col - window, 1, size(spot_img_rem, 2))
    c1 = clamp(pred_col + window, 1, size(spot_img_rem, 2))

    big_roi = spot_img_rem[r0:r1, c0:c1]

    # ── Centroid refinement: find the true center within this window ──────
    # Use intensity-weighted centroid (center of mass), which is more robust
    # than argmax for noisy PSFs and gives sub-pixel accuracy
    total_intensity = sum(big_roi)

    if total_intensity < 1e-6
        # Degenerate: no signal, keep affine prediction as fallback
        push!(true_centers, (Float64(pred_col), Float64(pred_row)))
        # Still extract a small centered ROI
        small_half = 5
        rs = clamp(pred_row - small_half, 1, size(spot_img_rem, 1))
        re = clamp(pred_row + small_half, 1, size(spot_img_rem, 1))
        cs = clamp(pred_col - small_half, 1, size(spot_img_rem, 2))
        ce = clamp(pred_col + small_half, 1, size(spot_img_rem, 2))
        push!(rois_centered, spot_img_rem[rs:re, cs:ce])
        continue
    end

    # Weighted centroid in ROI-local coordinates (1-based)
    rows_roi = r0:r1
    cols_roi = c0:c1

    centroid_row = 0.0
    centroid_col = 0.0
    for (ri, r) in enumerate(rows_roi)
        for (ci, c) in enumerate(cols_roi)
            w = max(0.0, spot_img_rem[r, c])
            centroid_row += w * r
            centroid_col += w * c
        end
    end
    centroid_row /= total_intensity
    centroid_col /= total_intensity

    push!(true_centers, (centroid_col, centroid_row))  # (col, row) convention

    # ── Re-extract a tightly centered ROI around the TRUE centroid ────────
    small_half = 5
    tc_row = round(Int, centroid_row)
    tc_col = round(Int, centroid_col)

    rs = clamp(tc_row - small_half, 1, size(spot_img_rem, 1))
    re = clamp(tc_row + small_half, 1, size(spot_img_rem, 1))
    cs = clamp(tc_col - small_half, 1, size(spot_img_rem, 2))
    ce = clamp(tc_col + small_half, 1, size(spot_img_rem, 2))

    push!(rois_centered, spot_img_rem[rs:re, cs:ce])
end

# ── Print: predicted vs true center for each spot ─────────────────────────
println("\nAffine prediction vs centroid-refined true center:")
println("  Spot  |  pred(row,col)  |  true(row,col)  |  offset(row,col)")
for i in eachindex(predicted_centers)
    pr, pc = predicted_centers_rc[i]
    tc, tr = true_centers[i]   # (col, row) stored above
    tr_i = round(tr, digits=1)
    tc_i = round(tc, digits=1)
    Δr = tr_i - pr
    Δc = tc_i - pc
    println("  $i     |  ($pr, $pc)     |  ($tr_i, $tc_i)    |  ($Δr, $Δc)")
end

# ── Visualization: spots now properly centered ─────────────────────────────
fig = Figure(size=(1200, 600))
for i in eachindex(rois_centered)
    row_panel = ceil(Int, i / 5)
    col_panel = mod1(i, 5)

    ax = CM.Axis(fig[row_panel, col_panel],
                  title = "Spot $i",
                  aspect = DataAspect())

    # Do NOT transpose here — the ROI is already [row, col] and
    # we want row on the vertical axis (y) increasing downward
    # which is standard image convention. heatmap maps dim1→x, dim2→y
    # so we DO need the transpose to show it as an image:
    heatmap!(ax, rois_centered[i]',
             colormap = :inferno)
end
display(fig)

# ── Full image overlay: predicted (cyan) vs true centroid (yellow) ─────────
fig2 = Figure(size=(900, 700))
ax2 = CM.Axis(fig2[1,1],
               yreversed = true,
               title = "Diffraction test: predicted (cyan) vs true centroid (yellow)")
heatmap!(ax2, spot_img_rem')

# Predicted centers from affine
scatter!(ax2,
         [p[1] for p in predicted_centers],   # xc = col
         [p[2] for p in predicted_centers],   # yc = row
         color = :cyan, markersize = 14,
         strokecolor = :black, strokewidth = 1,
         label = "affine prediction")

# True centroid centers
scatter!(ax2,
         [p[1] for p in true_centers],    # col
         [p[2] for p in true_centers],    # row
         color = :yellow, markersize = 10,
         marker = :cross, strokewidth = 2,
         label = "centroid refined")

axislegend(ax2)
display(fig2)

#--------------#---------------#----------------#------------#

# ==============================================================================
# Part 1: Fit 2D Gaussian to each of the 10 spots and extract σx, σy
# ==============================================================================
 
"""
    fit_2d_gaussian(roi)
 
Fit a 2D Gaussian to a small ROI (already background-subtracted).
Uses least-squares on the log-transformed image (linearizes the Gaussian),
which is fast and robust for well-formed PSF spots.
 
Returns (σ_col, σ_row, amplitude, offset) where σ_col and σ_row
are the 1/e² half-widths in column and row directions respectively,
in units of pixels.
 
Coordinate convention: ROI index [row, col], 1-based.
"""
function fit_2d_gaussian(roi::Matrix{<:Real})
    nr, nc = size(roi)
    
    # Centre of the ROI in 1-based pixel coords
    row_c = (nr + 1) / 2.0
    col_c = (nc + 1) / 2.0
    
    # Use intensity-weighted centroid as a better centre estimate
    total = sum(max.(roi, 0.0))
    if total < 1e-9
        return (σ_col=NaN, σ_row=NaN, amplitude=NaN, offset=NaN,
                centroid_col=col_c, centroid_row=row_c)
    end
    
    wrow = sum((1:nr) .* sum(max.(roi, 0.0), dims=2)[:]) / total
    wcol = sum((1:nc) .* sum(max.(roi, 0.0), dims=1)[:]) / total
    
    # Compute weighted second moments (= σ²) directly
    # This is more robust than log-linearisation for noisy data
    σ²_row = 0.0
    σ²_col = 0.0
    for r in 1:nr, c in 1:nc
        w = max(roi[r, c], 0.0)
        σ²_row += w * (r - wrow)^2
        σ²_col += w * (c - wcol)^2
    end
    σ²_row /= total
    σ²_col /= total
    
    amplitude = maximum(roi)
    offset    = minimum(roi)
    
    return (σ_col    = sqrt(σ²_col),
            σ_row    = sqrt(σ²_row),
            amplitude = amplitude,
            offset    = offset,
            centroid_col = wcol,
            centroid_row = wrow)
end
 
 
# Fit all 10 spots
println("=== 2D Gaussian fits to the 10 diffraction spots ===")
println("  Spot | σ_col (px) | σ_row (px) | FWHM_col (px) | FWHM_row (px) | Amplitude")
 
gaussian_fits = []
σ_cols = Float64[]
σ_rows = Float64[]
 
for (i, roi) in enumerate(rois_centered)
    gfit = fit_2d_gaussian(Float64.(roi))
    push!(gaussian_fits, gfit)
    
    if !isnan(gfit.σ_col)
        push!(σ_cols, gfit.σ_col)
        push!(σ_rows, gfit.σ_row)
        fwhm_c = 2.355 * gfit.σ_col
        fwhm_r = 2.355 * gfit.σ_row
        println("offset: ", gfit.offset)
        @printf("  %4d | %9.2f  | %9.2f  | %12.2f  | %12.2f  | %.1f\n",
                i, gfit.σ_col, gfit.σ_row, fwhm_c, fwhm_r, gfit.amplitude)
    else
        println("  $i | FAILED (insufficient signal)")
    end
end
 
println("\n--- Summary across all spots ---")
@printf("  Median σ_col = %.2f px  (FWHM = %.2f px)\n",
        median(σ_cols), 2.355*median(σ_cols))
@printf("  Median σ_row = %.2f px  (FWHM = %.2f px)\n",
        median(σ_rows), 2.355*median(σ_rows))
@printf("  Std σ_col    = %.2f px  (spot-to-spot variation)\n", std(σ_cols))
@printf("  Std σ_row    = %.2f px\n", std(σ_rows))
 
pixel_size_um = 3.45   # CS165MU1 pixel pitch in µm
@printf("\n  In physical units (%.2f µm/px):\n", pixel_size_um)
@printf("  Median FWHM_col = %.2f µm\n", 2.355*median(σ_cols)*pixel_size_um)
@printf("  Median FWHM_row = %.2f µm\n", 2.355*median(σ_rows)*pixel_size_um)
 
 
# ==============================================================================
# Part 2: 3D surface visualisation of a representative spot + Gaussian model
# ==============================================================================
 
# Use the spot with the highest peak signal for the 3D plot
best_spot_i = argmax([isnan(g.amplitude) ? -Inf : g.amplitude for g in gaussian_fits])
roi_3d      = Float64.(rois_centered[best_spot_i])
gfit_3d     = gaussian_fits[best_spot_i]
nr, nc      = size(roi_3d)
 
fig_3d = Figure(size=(1200, 500))
 
# ── Left: measured PSF as 3D surface ──────────────────────────────────────────
ax_meas = CM.Axis3(fig_3d[1, 1],
    title     = "Measured PSF — Spot $best_spot_i",
    xlabel    = "Column offset (px)",
    ylabel    = "Row offset (px)",
    zlabel    = "Intensity (ADU)",
    azimuth   = π/4,
    elevation = π/6)
 
col_axis = collect(1:nc) .- gfit_3d.centroid_col
row_axis = collect(1:nr) .- gfit_3d.centroid_row
 
surface!(ax_meas, col_axis, row_axis, roi_3d',
         colormap = :inferno)
 
# ── Right: fitted 2D Gaussian model ────────────────────────────────────────────
ax_model = CM.Axis3(fig_3d[1, 2],
    title     = "Fitted Gaussian model (σ_col=$(round(gfit_3d.σ_col,digits=2)) px, σ_row=$(round(gfit_3d.σ_row,digits=2)) px)",
    xlabel    = "Column offset (px)",
    ylabel    = "Row offset (px)",
    zlabel    = "Intensity (ADU)",
    azimuth   = π/4,
    elevation = π/6)
 
gauss_model = [gfit_3d.amplitude *
               exp(-0.5 * ((c / gfit_3d.σ_col)^2 + (r / gfit_3d.σ_row)^2))
               for r in row_axis, c in col_axis]
 
surface!(ax_model, col_axis, row_axis, gauss_model',
         colormap = :inferno)
 
display(fig_3d)
 
 
# ==============================================================================
# Part 3: σ distribution across the 10 spots
# ==============================================================================
 
fig_sigma = Figure(size=(700, 350))
ax_s = CM.Axis(fig_sigma[1, 1],
    title  = "PSF width distribution across 10 spots",
    xlabel = "Spot index",
    ylabel = "σ (px)")
 
scatter!(ax_s, 1:length(σ_cols), σ_cols,
         color=:dodgerblue, markersize=12, label="σ_col")
scatter!(ax_s, 1:length(σ_rows), σ_rows,
         color=:crimson, markersize=12, marker=:diamond, label="σ_row")
 
hlines!(ax_s, [median(σ_cols)], color=:dodgerblue, linestyle=:dash, linewidth=1)
hlines!(ax_s, [median(σ_rows)], color=:crimson,    linestyle=:dash, linewidth=1)
 
axislegend(ax_s)
display(fig_sigma)
 
 
# ==============================================================================
# Part 4: Gaussian-weighted intensity sampling (improved Method C)
#
# Instead of a flat 2×2 box average (Methods A/B), weight each camera pixel
# by the Gaussian PSF profile evaluated at its distance from the predicted
# SLM pixel centre.  This is the matched-filter estimate and gives the
# maximum-likelihood intensity reading for a Gaussian PSF.
#
# The kernel is pre-computed once using the median σ measured above,
# so it adds negligible runtime overhead vs Method A/B.
# ==============================================================================
 
# Use median σ across spots as the best single estimate of PSF width
σ_col_est = median(σ_cols)
σ_row_est = median(σ_rows)
 
# Half-width of the sampling kernel in pixels.
# 3σ captures 99.7% of a Gaussian's energy; round up to nearest integer.
kernel_half_col = ceil(Int, 3 * σ_col_est)
kernel_half_row = ceil(Int, 3 * σ_row_est)
 
println("\n=== Gaussian-weighted sampling kernel ===")
@printf("  σ_col = %.2f px  →  kernel half-width col = %d px\n",
        σ_col_est, kernel_half_col)
@printf("  σ_row = %.2f px  →  kernel half-width row = %d px\n",
        σ_row_est, kernel_half_row)
println("  Kernel footprint: $(2*kernel_half_row+1) × $(2*kernel_half_col+1) pixels")
 
"""
    gaussian_weighted_sample(cam_img, xc, yc, σ_col, σ_row,
                              half_col, half_row)
 
Sample the camera image at floating-point position (xc [col], yc [row])
using a Gaussian-weighted average over the surrounding kernel_half window.
Returns the weighted mean intensity (a scalar Float32).
 
This replaces the flat 2×2 box used in Methods A and B with a kernel that
matches the actual PSF shape, giving a more accurate and lower-noise
estimate of the true intensity at each SLM pixel's conjugate location.
"""
function gaussian_weighted_sample(cam_img::Matrix{Float32},
                                    xc::Float64, yc::Float64,
                                    σ_col::Float64, σ_row::Float64,
                                    half_col::Int, half_row::Int)
 
    cam_rows, cam_cols = size(cam_img)
 
    # Integer centre
    col0 = round(Int, xc)
    row0 = round(Int, yc)
 
    # Sub-pixel offset
    δcol = xc - col0
    δrow = yc - row0
 
    weighted_sum = 0.0f0
    weight_total = 0.0f0
 
    for dr in -half_row:half_row
        r = row0 + dr
        (r < 1 || r > cam_rows) && continue
 
        for dc in -half_col:half_col
            c = col0 + dc
            (c < 1 || c > cam_cols) && continue
 
            # Distance from the true (sub-pixel) centre
            d_col = (dc - δcol) / σ_col
            d_row = (dr - δrow) / σ_row
 
            w = exp((-0.5f0) * Float32(d_col^2 + d_row^2))
 
            weighted_sum += w * cam_img[r, c]
            weight_total += w
        end
    end
 
    return weight_total > (1.0f-9) ? weighted_sum / weight_total : (0.0f0)
end
 
 
# ==============================================================================
# Part 5: Re-run the 256-step sweep using Gaussian-weighted sampling (Method C)
#         Drop-in replacement for the Method A / Method B loops
# ==============================================================================
 
# Pre-compute floating-point camera coordinates for every SLM pixel (once)
xc_map_f64 = zeros(Float64, num_slm_x, num_slm_y)
yc_map_f64 = zeros(Float64, num_slm_x, num_slm_y)
 
for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)
        xc_map_f64[xi, yi] =
            affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc_map_f64[xi, yi] =
            affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    end
end
 
intensity_cube_C = zeros(Float32, num_slm_x, num_slm_y, 256)
 
println("\nInitializing 256-Step Sweep — Method C (Gaussian-weighted)...")
 
for v in 0:255
    slm.phase = fill(v / 255.0, 1024, 1024)
    Meadowlark.writesingleimage(slm)
 
    cam_frame = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end
    cam_img = Float32.(cam_frame)
 
    for xi in 1:num_slm_x
        for yi in 1:num_slm_y
            intensity_cube_C[xi, yi, v+1] = gaussian_weighted_sample(
                cam_img,
                Float64(xc_map_f64[xi, yi]),
                Float64(yc_map_f64[xi, yi]),
                σ_col_est, σ_row_est,
                kernel_half_col, kernel_half_row)
        end
    end
 
    v % 32 == 0 && println("  Sweep progress: $v/255")
end
 
println("Method C sweep complete. Cube shape: ", size(intensity_cube_C))
 
 
# ==============================================================================
# Part 6: Side-by-side comparison of Method A vs Method C on the same pixel
# ==============================================================================
 
mid_xi = num_slm_x ÷ 2
mid_yi = num_slm_y ÷ 2
 
curve_A = Float64.(intensity_cube[mid_xi, mid_yi, :])
curve_C = Float64.(intensity_cube_C[mid_xi, mid_yi, :])
 
# Normalise both to their own maximum for a fair shape comparison
curve_A_norm = curve_A ./ maximum(curve_A)
curve_C_norm = curve_C ./ maximum(curve_C)
 
fig_cmp = Figure(size=(1000, 450))
ax_cmp  = CM.Axis(fig_cmp[1, 1],
    title  = "Method A (flat 2×2) vs Method C (Gaussian-weighted) — SLM ($(slm_x_range[mid_xi]), $(slm_y_range[mid_yi]))",
    xlabel = "Voltage Step (0–255)",
    ylabel = "Normalised Intensity")
 
lines!(ax_cmp, 0:255, curve_A_norm,
       color=:dodgerblue, linewidth=2, label="Method A — flat 2×2")
lines!(ax_cmp, 0:255, curve_C_norm,
       color=:crimson,    linewidth=2, label="Method C — Gaussian-weighted")
axislegend(ax_cmp, position=:rb)
display(fig_cmp)
 
# Noise proxy: std of the residual after smoothing (lower = less noise in the curve)
function curve_noise(y; window=5)
    n = length(y)
    smoothed = [mean(y[max(1,i-window÷2):min(n,i+window÷2)]) for i in 1:n]
    return std(y .- smoothed)
end
 
noise_A = curve_noise(curve_A_norm)
noise_C = curve_noise(curve_C_norm)
@printf("\nNoise proxy (Method A) = %.5f\n", noise_A)
@printf("Noise proxy (Method C) = %.5f\n", noise_C)
@printf("Improvement factor     = %.2f×\n", noise_A / noise_C)








































# ==============================================================================
#9. Split the 256-step curve into 4 distinct voltage intervals separated by these calculated extrema positions. 
# Scale the intensity values inside each individual segment smoothly to a range of $[0.0, 1.0]$.
# Fig 2.b
# ==============================================================================

segmented_data = [Vector{Vector{Float32}}() for xi in 1:Nx, yi in 1:Ny]

println("Beginning Step 9: Slicing and Normalizing Phase Stroke Segments via Smoothed Data...")

segment_colors = [:deepskyblue, :darkorange, :crimson, :forestgreen]
segment_labels = [
    "Segment 1 (Min1 → Max1)",
    "Segment 2 (Max1 → Min2)",
    "Segment 3 (Min2 → Max2)",
    "Segment 4 (Max2 → Min3)"
]

mid_xi = round(Int, Nx / 2)
mid_yi = round(Int, Ny / 2)
actual_slm_x = slm_x_range[mid_xi]
actual_slm_y = slm_y_range[mid_yi]

fig_seg = Figure(size=(850, 500))
ax_seg = CM.Axis(
    fig_seg[1, 1],
    title = "Normalized 4-Segment Phase Stroke Overlay (Smoothed)\nSLM Pixel ($actual_slm_x, $actual_slm_y)",
    xlabel = "Normalized Voltage Axis (V / V_max)",
    ylabel = "Normalized Camera Intensity",
    aspect = 1.0
)
xlims!(ax_seg, -0.02, 1.02)
ylims!(ax_seg, -0.02, 1.02)

for xi in 1:Nx
    for yi in 1:Ny
        
        # 1. Generate the smoothed profile for segment extraction
        raw_profile = intensity_cube[xi, yi, :]
        #smoothed_profile = smooth_profile(raw_profile, 2) # Matches Step 8 filter
        #trying w/o smoothing, gives pretty much same quality data
        smoothed_profile = raw_profile
        
        if pixel_quality_map[xi, yi] == 4
            
            boundaries = [
                min1_map[xi, yi],
                max1_map[xi, yi],
                min2_map[xi, yi],
                max2_map[xi, yi],
                min3_map[xi, yi]
            ]
            
            for s in 1:4
                # Safely convert Float64 extrema to nearest Int for index slicing
                start_idx_int = round(Int, boundaries[s])
                end_idx_int   = round(Int, boundaries[s+1])
                
                # Slice the SMOOTHED intensities
                seg_intensities = smoothed_profile[start_idx_int:end_idx_int]
                seg_voltages    = Float32.( (start_idx_int-1):(end_idx_int-1) )
                
                # Masterstroke: Normalize X-axis using the exact parabolic floating coordinates
                v_min = Float32(boundaries[s] - 1)
                v_max = Float32(boundaries[s+1] - 1)
                norm_v = (seg_voltages .- v_min) ./ (v_max - v_min)
                
                # Normalize Y-axis on the smoothed data envelope
                i_min, i_max = extrema(seg_intensities)
                norm_i = (i_max > i_min) ? (seg_intensities .- i_min) ./ (i_max - i_min) : zeros(Float32, length(seg_intensities))
                
                push!(segmented_data[xi, yi], norm_i)
                
                # Plot test target pixel live
                if xi == mid_xi && yi == mid_yi
                    lines!(ax_seg, norm_v, norm_i, color = segment_colors[s], linewidth = 2.5, label = segment_labels[s])
                    scatter!(ax_seg, norm_v, norm_i, color = segment_colors[s], markersize = 6)
                end
            end
            
        elseif pixel_quality_map[xi, yi] == 2
            # 2π Fallback handling using smoothed profiles
            boundaries = [min1_map[xi, yi], max1_map[xi, yi], min2_map[xi, yi]]
            
            for s in 1:2
                start_idx_int = round(Int, boundaries[s])
                end_idx_int   = round(Int, boundaries[s+1])
                
                seg_intensities = smoothed_profile[start_idx_int:end_idx_int]
                seg_voltages    = Float32.( (start_idx_int-1):(end_idx_int-1) )
                
                v_min = Float32(boundaries[s] - 1)
                v_max = Float32(boundaries[s+1] - 1)
                norm_v = (seg_voltages .- v_min) ./ (v_max - v_min)
                
                i_min, i_max = extrema(seg_intensities)
                norm_i = (i_max > i_min) ? (seg_intensities .- i_min) ./ (i_max - i_min) : zeros(Float32, length(seg_intensities))
                
                push!(segmented_data[xi, yi], norm_i)
            end
        end
        
    end
end

axislegend(ax_seg, position = :rt, framevisible = true, bgcolor = (:white, 0.85))
display(fig_seg)

println("Step 9 Complete! Normalized phase curves map elegantly to a clean template.")








# ==============================================================================
# 10. Phase Inversion & Monotonic Phase Unrolling
# ==============================================================================

# Structure to hold the full continuous phase curve per pixel 
# We track the raw voltage steps and their corresponding absolute unrolled phase
unrolled_phase_data = [
    Vector{NamedTuple{(:Point, :V, :Phi), Tuple{Int, Float32, Float32}}}()
    for xi in 1:Nx, yi in 1:Ny
]

# Storage for the overlay plot data (local phase accrued vs normalized voltage)
overlay_phase_curves = [Vector{Vector{Float32}}() for xi in 1:Nx, yi in 1:Ny]
overlay_voltage_curves = [Vector{Vector{Float32}}() for xi in 1:Nx, yi in 1:Ny]

println("Beginning Step 10: Transforming Intensity to Phase Space...")

# Loop over every pixel in the ROI
for xi in 1:Nx
    for yi in 1:Ny
        
        # Only process high-quality 4π phase stroke pixels
        if pixel_quality_map[xi, yi] == 4
            
            # Retrieve precise parabolic boundaries from Step 8
            boundaries = [
                min1_map[xi, yi],
                max1_map[xi, yi],
                min2_map[xi, yi],
                max2_map[xi, yi],
                min3_map[xi, yi]
            ]
            
            raw_profile = intensity_cube[xi, yi, :]
            smoothed_profile = smooth_profile(raw_profile, 2)
            
            # Phase baseline offsets for each segment to unroll continuously up to 4π
            phase_baselines = [0.0f0, Float32(π), Float32(2π), Float32(3π)]
            
            for s in 1:4
                start_idx_int = round(Int, boundaries[s])
                end_idx_int   = round(Int, boundaries[s+1])
                
                # Slice segment intensities and setup matching voltage tracking arrays
                seg_intensities = smoothed_profile[start_idx_int:end_idx_int]
                seg_voltages    = Float32.( (start_idx_int-1):(end_idx_int-1) )
                
                # Normalize X-axis using floating parabolic coordinates
                v_min = Float32(boundaries[s] - 1)
                v_max = Float32(boundaries[s+1] - 1)
                norm_v = (seg_voltages .- v_min) ./ (v_max - v_min)
                
                # Normalize Y-axis intensity strictly to [0.0, 1.0]
                i_min, i_max = extrema(seg_intensities)
                norm_i = (i_max > i_min) ? (seg_intensities .- i_min) ./ (i_max - i_min) : zeros(Float32, length(seg_intensities))
                
                # Apply inverse interferometric equations based on segment direction
                local_phi = zeros(Float32, length(norm_i))
                if s == 1 || s == 3
                    # Rising segments: Min to Max
                    local_phi .= 2.0f0 .* asin.(sqrt.(norm_i))
                else
                    # Falling segments: Max to Min
                    local_phi .= 2.0f0 .* acos.(sqrt.(norm_i))
                end
                
                # Compute absolute unrolled phase values
                abs_phi = local_phi .+ phase_baselines[s]
                
                # Save data for the local overlay plot
                push!(overlay_phase_curves[xi, yi], local_phi)
                push!(overlay_voltage_curves[xi, yi], norm_v)
                
                # Stream into global continuous data log for LUT creation
                for idx in 1:length(seg_voltages)
                    push!(unrolled_phase_data[xi, yi], (
                        Point = start_idx_int + idx - 1,
                        V     = seg_voltages[idx],
                        Phi   = abs_phi[idx]
                    ))
                end
            end
        end
        
    end
end

println("Phase inversion complete. Generating diagnostic plots...")

# ==============================================================================
# Visualization: 2-Panel Diagnostic Figure
# ==============================================================================
mid_xi = round(Int, Nx / 2)
mid_yi = round(Int, Ny / 2)
actual_slm_x = slm_x_range[mid_xi]
actual_slm_y = slm_y_range[mid_yi]

fig_phase = Figure(size=(1200, 500))

# Panel A: Segment Overlay (Local Phase vs Normalized Voltage)
ax_overlay = CM.Axis(
    fig_phase[1, 1],
    title = "Normalized Segment Overlay: Phase vs Voltage\nSLM Pixel ($actual_slm_x, $actual_slm_y)",
    xlabel = "Normalized Voltage Axis (V / V_max)",
    ylabel = "Local Phase Accrued (Radians)",
    yticks = (0:π/4:π, ["0", "π/4", "π/2", "3π/4", "π"])
)

# Panel B: Continuous Unrolled Phase Curve
ax_unrolled = CM.Axis(
    fig_phase[1, 2],
    title = "Continuous Absolute Phase Evolution (4π Stroke)\nSLM Pixel ($actual_slm_x, $actual_slm_y)",
    xlabel = "Raw Voltage Step (0 - 255)",
    ylabel = "Absolute Phase (Radians)",
    yticks = (0:π:4π, ["0", "1π", "2π", "3π", "4π"])
)

segment_colors = [:deepskyblue, :darkorange, :crimson, :forestgreen]
segment_labels = ["Seg 1 (0 → π)", "Seg 2 (π → 2π)", "Seg 3 (2π → 3π)", "Seg 4 (3π → 4π)"]

# Populate Panel A: Overlay curves
for s in 1:4
    v_curve = overlay_voltage_curves[mid_xi, mid_yi][s]
    p_curve = overlay_phase_curves[mid_xi, mid_yi][s]
    
    lines!(ax_overlay, v_curve, p_curve, color = segment_colors[s], linewidth = 2.5, label = segment_labels[s])
    scatter!(ax_overlay, v_curve, p_curve, color = segment_colors[s], markersize = 5)
end
axislegend(ax_overlay, position = :rb, framevisible = true)

# Populate Panel B: Continuous absolute curve
target_unrolled = unrolled_phase_data[mid_xi, mid_yi]
v_raw = [pt.V for pt in target_unrolled]
phi_abs = [pt.Phi for pt in target_unrolled]

# Sort tracking arrays chronologically by voltage to handle any overlapping slice indices gracefully
p = sortperm(v_raw)
v_raw_sorted = v_raw[p]
phi_abs_sorted = phi_abs[p]

lines!(ax_unrolled, v_raw_sorted, phi_abs_sorted, color = :purple, linewidth = 3)
scatter!(ax_unrolled, v_raw_sorted, phi_abs_sorted, color = :black, markersize = 4)

display(fig_phase)
println("Step 10 Diagnostic Figures Rendered!")



# ==============================================================================
#11. Invert to create the Look-Up Table
# Fig 2.d
# ==============================================================================
println("Beginning Step 11: Inverting Multi-Tiered Phase Curves into Unified 8-bit Maps...")

# Define our global target phase axis: 256 steps spanning a full 4π cycle
target_phase_steps = Float32.(range(0.0, stop=4π, length=256))

# Initialize the 3D Regional Matrix (Nx x Ny x 256 Phase Graylevels)
regional_lut_matrix = zeros(UInt8, Nx, Ny, 256)

count_4pi = 0
count_2pi = 0
count_fallback = 0

for xi in 1:Nx
    for yi in 1:Ny
        
        quality = pixel_quality_map[xi, yi]
        
        if quality == 4 || quality == 2
            pixel_data = unrolled_phase_data[xi, yi]
            
            # Sort chronologically by absolute tracking phase
            p = sortperm([pt.Phi for pt in pixel_data])
            sorted_data = pixel_data[p]
            
            v_raw   = [pt.V for pt in sorted_data]
            phi_abs = [pt.Phi for pt in sorted_data]
            phi_start = phi_abs[1]
            
            for grey in 1:256
                # Determine the nominal target phase relative to this pixel's start point
                rel_target = target_phase_steps[grey]
                
                # TIERED WRAPPING: If this pixel only supports a 2π stroke, wrap target phase inside the LUT
                if quality == 2
                    rel_target = mod(rel_target, 2π)
                end
                
                target_phi = phi_start + rel_target
                
                # Local linear interpolation inside the pixel's calibrated phase sweep
                if target_phi <= phi_abs[1]
                    target_v_step = v_raw[1]
                elseif target_phi >= phi_abs[end]
                    target_v_step = v_raw[end]
                else
                    idx = findfirst(p -> p >= target_phi, phi_abs)
                    idx = (idx === nothing) ? length(phi_abs) : idx
                    idx_low = max(1, idx - 1)
                    
                    d_phi = phi_abs[idx] - phi_abs[idx_low]
                    weight = (d_phi > 0.0f0) ? (target_phi - phi_abs[idx_low]) / d_phi : 0.0f0
                    
                    target_v_step = v_raw[idx_low] + weight * (v_raw[idx] - v_raw[idx_low])
                end
                
                # Save directly as 8-bit unsigned integer voltage steps
                regional_lut_matrix[xi, yi, grey] = UInt8(clamp(round(Int, target_v_step), 0, 255))
            end
            
            if quality == 4; count_4pi += 1; else; count_2pi += 1; end
            
        else
            # FALLBACK: Assign nominal uncalibrated drive scale assuming 4π response over 0-255
            for grey in 1:256
                regional_lut_matrix[xi, yi, grey] = UInt8(grey - 1)
            end
            count_fallback += 1
        end
        
    end
end

datadir = "W:\\Projects\\CU-MINFLUX\\SLM calibration\\"
mkpath(datadir) 
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

save_path = joinpath(
    datadir,
    "$(ROISize)x$(ROISize)_center_$(slm_x_range[mid_xi])_$(slm_y_range[mid_yi])_$timestamp.jld2"
)


println("Saving spatial lookup matrix to: $save_path")
@save save_path regional_lut_matrix slm_x_range slm_y_range pixel_quality_map

println("Step 11 Complete!")
println(" -> 4π Calibrated Pixels: $count_4pi")
println(" -> 2π Wrapped Calibrated Pixels: $count_2pi")
println(" -> Fallback/Nominal Pixels: $count_fallback")





# ================================================================================
function genDiagnosticQuadrantGrid(center_x=514, center_y=561, box_size=480)
    N = 1024 # Native SLM resolution
    phaseGrid = zeros(Float32, N, N)
    
    # Calculate bounding box bounds
    half_size = div(box_size, 2)
    x_start = center_x - half_size + 1
    x_end   = center_x + half_size
    y_start = center_y - half_size + 1
    y_end   = center_y + half_size
    
    # Phase values to test different stroke areas of the LUT
    p1 = 0.0f0          # Top-Left
    p2 = Float32(π / 2)  # Top-Right
    p3 = Float32(π)      # Bottom-Left
    p4 = Float32(3π / 2) # Bottom-Right

    for y in y_start:y_end
        is_bottom = y > center_y
        for x in x_start:x_end
            is_right = x > center_x
            
            if !is_bottom && !is_right
                phaseGrid[x, y] = p1 # Top-Left
            elseif !is_bottom && is_right
                phaseGrid[x, y] = p2 # Top-Right
            elseif is_bottom && !is_right
                phaseGrid[x, y] = p3 # Bottom-Left
            else
                phaseGrid[x, y] = p4 # Bottom-Right
            end
        end
    end
    
    return phaseGrid, (x_start:x_end, y_start:y_end)
end


phase_pattern, (x_range, y_range) =
    genDiagnosticQuadrantGrid(center_x, center_y, 480)

raw_frame = zeros(UInt8, 1024, 1024)

for y in y_range
    for x in x_range

        p = phase_pattern[x,y]

        raw_frame[y,x] =
            p ≈ 0      ? UInt8(0)   :
            p ≈ π/2    ? UInt8(64)  :
            p ≈ π      ? UInt8(128) :
                          UInt8(192)
    end
end

slm.phase = Float64.(raw_frame) ./ 255
Meadowlark.writesingleimage(slm)



testing_image = try
    ThorCamCSC.capture(test_cam)
finally
    ThorCamCSC.disarmcamera(test_cam)
end

imshow(Float64.(testing_image)) #


















# ==============================================================================
# Graph a particular phase function og verion
# ==============================================================================

function prepare_hardware_frame(target_phase_matrix::Matrix{Float32}, regional_lut::Array{UInt8, 3}, slm_x_range, slm_y_range; native_width=1024, native_height=1024)
    
    # 1. Allocate full native display frame buffer
    full_frame = zeros(UInt8, native_width, native_height)
    
    # 2. Pre-populate the global frame with baseline nominal linear phase mapping.
    # This keeps the uncalibrated frame borders safe and predictable.
    for x in 1:native_width
        for y in 1:native_height
            # Wrap nominal phase to 4π to match our global pipeline range
            wrapped_nominal = mod(target_phase_matrix[x, y], 4π)
            full_frame[x, y] = UInt8(clamp(round(Int, (wrapped_nominal / 4π) * 255.0), 0, 255))
        end
    end
    
    # 3. Overlay the high-precision calibrated region
    Nx, Ny = length(slm_x_range), length(slm_y_range)
    for xi in 1:Nx
        for yi in 1:Ny
            slm_x = slm_x_range[xi]
            slm_y = slm_y_range[yi]
            
            # Universally wrap incoming target phase to the 4π pipeline scale
            wrapped_phase = mod(target_phase_matrix[slm_x, slm_y], 4π)
            
            # Quantize phase value into index 1-256
            phase_idx = clamp(round(Int, (wrapped_phase / 4π) * 255.0) + 1, 1, 256)
            
            # Write out the customized calibrated byte directly
            full_frame[slm_x, slm_y] = regional_lut[xi, yi, phase_idx]
        end
    end
    
    return full_frame
end



#Did not work...
# ==============================================================================
# 1. Function to Generate a Pure Vortex (Donut) Phase Pattern
# ==============================================================================
function generate_vortex_target(native_width::Int, native_height::Int, center_x::Real, center_y::Real; charge::Int=1)
    # Pre-allocate the matrix in radians
    target_phase = zeros(Float32, native_width, native_height)
    
    for x in 1:native_width
        for y in 1:native_height
            dx = x - center_x
            dy = y - center_y
            
            if dx == 0 && dy == 0
                target_phase[x, y] = 0.0f0
            else
                # atan2 returns values from -π to +π. 
                # We add π to shift the range smoothly to [0, 2π].
                angle = atan(dy, dx) + π
                
                # Multiply by the topological charge (standard MINFLUX is charge=1)
                target_phase[x, y] = Float32(charge * angle)
            end
        end
    end
    return target_phase
end


# 1. Determine the exact pixel center of your 475x475 calibrated ROI
mid_xi = div(length(slm_x_range), 2) + 1
mid_yi = div(length(slm_y_range), 2) + 1

center_x = slm_x_range[mid_xi]
center_y = slm_y_range[mid_yi]

println("Generating a donut phase pattern centered at SLM pixel: ($center_x, $center_y)")

# 2. Generate the pure target phase profile (1024 x 1024 matrix in radians)
native_w, native_h = 1024, 1024
donut_target_phase = generate_vortex_target(native_w, native_h, center_x, center_y, charge=1)

# 3. Process the target phase map through your high-precision multi-tiered regional LUT
# This function applies your 4π/2π/Nominal calibrations and pads the borders.
calibrated_hardware_frame = prepare_hardware_frame(
    donut_target_phase, 
    regional_lut_matrix, 
    slm_x_range, 
    slm_y_range; 
    native_width=native_w, 
    native_height=native_h
)

calibrated_hardware_frame_f64 = Float64.(calibrated_hardware_frame) ./ 255.0

slm.phase = calibrated_hardware_frame_f64

Meadowlark.writesingleimage(slm) # returns 1 for success




uncalibrated_vortex = zeros(UInt8, 1024, 1024)
for x in 1:1024
    for y in 1:1024
        dx = x - center_x
        dy = y - center_y
        if dx != 0 || dy != 0
            angle = atan(dy, dx) + π  # Shunted to [0, 2π]
            # Map [0, 2π] linearly onto [0, 255] bytes
            uncalibrated_vortex[x, y] = UInt8(clamp(round(Int, (angle / 2π) * 255.0), 0, 255))
        end
    end
end

# 2. Push directly to the SLM
slm.phase = Float64.(uncalibrated_vortex) ./ 255.0
Meadowlark.writesingleimage(slm)




testing_image = try
    ThorCamCSC.capture(test_cam)
finally
    ThorCamCSC.disarmcamera(test_cam)
end

imshow(Float64.(testing_image)) #









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
cx = 514#514
cy = 561#561
#Note to self (512, 512)_SLM maps to (713, 433)_Camera. 
#and (514, 561)_SLM maps to (720, 540)_Camera aka the center of the camera. 
#so the scaling factor is $\frac{\sqrt{(720-713)^2 + (540-433)^2}}{\sqrt{(561-512)^2 + (514-512)^2}} \approx 2.19$, 
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










# Built-in library for handling time delays

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