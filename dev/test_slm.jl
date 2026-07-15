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
using Random
using StatsBase


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

imshow(Float64.(grid_raw)) 

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
imshow(Float64.(bg_raw)) 


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
# this step takes a ton of time...
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







# ==============================================================================
#12. Validate LUT by executing both uncalibrated and calibrated hardware sweeps
# Fig 2.E
# ==============================================================================


#Used to prepare the full 1024x1024 frame for the SLM, overlaying the calibrated region from the LUT onto a baseline nominal phase mapping.
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



"""
Executes both uncalibrated (voltage-linear) and calibrated (phase-linear) 
hardware sweeps, sampling 20 random golden pixels via their affine-mapped 
camera coordinates and Gaussian PSF profiles to generate Figure E.
"""
function run_figure_E_validation(regional_lut_matrix, pixel_quality_map, 
                                 min1_map, min3_map, affine_matrix,
                                 slm_x_range, slm_y_range, 
                                 σ_col_final, σ_row_final, phase_dark)
    
    Nx, Ny = length(slm_x_range), length(slm_y_range)
    kh_col = ceil(Int, 3 * σ_col_final)
    kh_row = ceil(Int, 3 * σ_row_final)

    # 1. Isolate and sample 20 random pixels exclusively from the golden 4π set
    golden_indices = findall(q -> q == 4, pixel_quality_map)
    if length(golden_indices) < 20
        error("Insufficient 4π pixels available in pixel_quality_map to sample 20 targets.")
    end
    
    rng = Random.MersenneTwister(42) # Fixed seed for tracking consistency
    sampled_indices = StatsBase.sample(rng, golden_indices, 20, replace=false)
    
    # Pre-calculate and cache the sub-pixel camera coordinates for our 20 targets
    target_metadata = []
    for idx in sampled_indices
        xi, yi = idx[1], idx[2]
        xs = slm_x_range[xi]
        ys = slm_y_range[yi]
        
        # Map through affine coordinate transform matrix
        xc = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
        
        push!(target_metadata, (xi=xi, yi=yi, xs=xs, ys=ys, xc=xc, yc=yc))
    end

    # 256 discrete execution steps mapping directly across the 4π profile
    num_steps = 256
    intensities_pre  = zeros(Float32, 20, num_steps)
    intensities_post = zeros(Float32, 20, num_steps)

    # =========================================================================
    # HARDWARE SWEEP 1: UNCALIBRATED (Locally Linear Grayvalue Spacing)
    # =========================================================================
    println("\nExecuting Uncalibrated Hardware Sweep (Linear Voltage Spacing)...")
    for v in 0:255
        # Frame assembly uses the row/column layout verified by your mask script [x, y]
        active_frame = copy(phase_dark)
        
        for yi in 1:Ny, xi in 1:Nx
            if pixel_quality_map[xi, yi] == 4
                v_start = min1_map[xi, yi]
                v_end   = min3_map[xi, yi]
                # Linearly space raw gray values between local extrema milestones
                g_val = v_start + (v / 255.0) * (v_end - v_start)
                
                xs = slm_x_range[xi]
                ys = slm_y_range[yi]
                active_frame[xs, ys] = clamp(round(Int, g_val), 0, 255) / 255.0
            end
        end
        
        # Push frame to SLM and capture hardware response
        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)
        
        cam_frame = try
            ThorCamCSC.capture(test_cam)
        finally
            ThorCamCSC.disarmcamera(test_cam)
        end
        cam_img = Float32.(cam_frame)
        
        # Query our 20 targets using their exact affine centers and Gaussian weights
        for (i, t) in enumerate(target_metadata)
            intensities_pre[i, v+1] = gaussian_weighted_sample(
                cam_img, t.xc, t.yc, σ_col_final, σ_row_final, kh_col, kh_row
            )
        end
        
        if v % 64 == 0; println("  Pre-sweep step $v/255 complete."); end
    end

    # =========================================================================
    # HARDWARE SWEEP 2: CALIBRATED (Phase-Linear Spacing via Regional LUT)
    # =========================================================================
    println("\nExecuting Calibrated Hardware Sweep (Linear Phase Spacing)...")
    for v in 0:255
        active_frame = copy(phase_dark)
        
        for yi in 1:Ny, xi in 1:Nx
            if pixel_quality_map[xi, yi] == 4
                # Space gray values non-linearly using the localized calibration curve
                g_val = regional_lut_matrix[xi, yi, v+1]
                
                xs = slm_x_range[xi]
                ys = slm_y_range[yi]
                active_frame[xs, ys] = g_val / 255.0
            end
        end
        
        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)
        
        cam_frame = try
            ThorCamCSC.capture(test_cam)
        finally
            ThorCamCSC.disarmcamera(test_cam)
        end
        cam_img = Float32.(cam_frame)
        
        for (i, t) in enumerate(target_metadata)
            intensities_post[i, v+1] = gaussian_weighted_sample(
                cam_img, t.xc, t.yc, σ_col_final, σ_row_final, kh_col, kh_row
            )
        end
        
        if v % 64 == 0; println("  Post-sweep step $v/255 complete."); end
    end

    # =========================================================================
    # VISUALIZATION GENERATION (CairoMakie)
    # =========================================================================
    fig = Figure(size = (1200, 550), font = "Arial")
    phase_axis = range(0.0, stop=4π, length=num_steps)
    
    ax_pre = CM.Axis(fig[1, 1], 
        title = "A) Pre-Calibration (Linear Voltage Steps)",
        xlabel = "Nominal Commanded Phase (Radians)",
        ylabel = "Normalized Intensity (Matched Filter)",
        xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"])
    )
    
    ax_post = CM.Axis(fig[1, 2], 
        title = "B) Post-Calibration (Linear Phase Steps via Regional LUT)",
        xlabel = "Nominal Commanded Phase (Radians)",
        ylabel = "Normalized Intensity (Matched Filter)",
        xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"])
    )
    
    #colors = Colors.DistinguishableQuantities.distinguishable_colors(20, [RGB(0.1, 0.1, 0.1)])
    colors = cgrad(:tab20, 20, categorical=true)

    for i in 1:20
        # Min-max normalization isolates the phase alignments from raw spatial illumination gradients
        curve_pre  = intensities_pre[i, :]
        curve_post = intensities_post[i, :]
        
        norm_pre  = (curve_pre  .- minimum(curve_pre))  ./ (maximum(curve_pre)  - minimum(curve_pre)  + 1e-6)
        norm_post = (curve_post .- minimum(curve_post)) ./ (maximum(curve_post) - minimum(curve_post) + 1e-6)
        
        lines!(ax_pre,  phase_axis, norm_pre,  color = colors[i], linewidth = 1.4)
        lines!(ax_post, phase_axis, norm_post, color = colors[i], linewidth = 1.8)
    end
    
    # Overlay phase milestone boundaries to visually prove peak/valley alignment
    for phase_milestone in 0:π:4π
        vlines!(ax_pre,  [phase_milestone], color = (:gray, 0.4), linestyle = :dash)
        vlines!(ax_post, [phase_milestone], color = (:red, 0.5),  linestyle = :dash)
    end
    
    save("figure_2E_validation.png", fig)
    println("\nValidation plot saved as 'figure_2E_validation.png'.")
    
    return fig
end

display(run_figure_E_validation(
    regional_lut_matrix, pixel_quality_map, 
    min1_map, min3_map, affine_matrix,
    slm_x_range, slm_y_range, 
    σ_col_final, σ_row_final, phase_dark
))



# ==============================================================================
#13. Validate LUT by analyzing wavefront error
# Fig 2.F
# ==============================================================================
# 1. CONFIGURATION & TIME TRACKING PARAMETERS
interval_minutes = 5      # Time between consecutive validation sweeps
total_hours      = 1/3     # Total observation window duration
num_steps        = 256     # Discrete phase steps per validation run (0 to 4π)

total_minutes = total_hours * 60  
time_points_mins = collect(0:interval_minutes:total_minutes)
num_time_points = length(time_points_mins)

Nx, Ny = length(slm_x_range), length(slm_y_range)
kh_col = ceil(Int, 3 * σ_col_final)
kh_row = ceil(Int, 3 * σ_row_final)

# Pre-compute floating-point camera coordinate mapping layout
xc_map_C = zeros(Float64, Nx, Ny)
yc_map_C = zeros(Float64, Nx, Ny)
for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)
        xc_map_C[xi, yi] = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc_map_C[xi, yi] = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    end
end

# Cache spatial intensity boundaries for absolute phase branch reconstruction
I_min1_map = zeros(Float32, Nx, Ny)
I_max1_map = zeros(Float32, Nx, Ny)
I_min2_map = zeros(Float32, Nx, Ny)
I_max2_map = zeros(Float32, Nx, Ny)
I_min3_map = zeros(Float32, Nx, Ny)

for yi in 1:Ny, xi in 1:Nx
    if pixel_quality_map[xi, yi] >= 2
        I_min1_map[xi, yi] = intensity_cube[xi, yi, round(Int, min1_map[xi, yi])]
        I_max1_map[xi, yi] = intensity_cube[xi, yi, round(Int, max1_map[xi, yi])]
        I_min2_map[xi, yi] = intensity_cube[xi, yi, round(Int, min2_map[xi, yi])]
    end
    if pixel_quality_map[xi, yi] == 4
        I_max2_map[xi, yi] = intensity_cube[xi, yi, round(Int, max2_map[xi, yi])]
        I_min3_map[xi, yi] = intensity_cube[xi, yi, round(Int, min3_map[xi, yi])]
    end
end

# Data Storage Matrices
phase_axis     = range(0.0f0, 4f0*π, length=num_steps)
baseline_rms   = zeros(Float32, num_steps)
baseline_dphi  = zeros(Float32, num_steps)
calib_rms_ts   = zeros(Float32, num_time_points, num_steps)
calib_dphi_ts  = zeros(Float32, num_time_points, num_steps)

# =========================================================================
# 2. STEP 13.A: EXECUTE UNCALIBRATED LINEAR VOLTAGE BASELINE SWEEP
# =========================================================================
println("\n=== Executing Uncalibrated Baseline Sweep (Linear Voltage Spacing) ===")
for v in 0:255
    phi_cmd = phase_axis[v+1]
    active_frame = copy(phase_dark)
    
    for yi in 1:Ny, xi in 1:Nx
        if pixel_quality_map[xi, yi] == 4
            v_start = min1_map[xi, yi]
            v_end   = min3_map[xi, yi]
            g_val   = v_start + (v / 255.0) * (v_end - v_start)
            xs, ys  = slm_x_range[xi], slm_y_range[yi]
            active_frame[xs, ys] = clamp(round(Int, g_val), 0, 255) / 255.0
        end
    end
    
    slm.phase = active_frame
    Meadowlark.writesingleimage(slm)
    cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
    cam_img = Float32.(cam_frame)
    
    valid_phases = Float32[]
    for yi in 1:Ny, xi in 1:Nx
        pixel_quality_map[xi, yi] != 4 && continue
        i_meas = gaussian_weighted_sample(cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], σ_col_final, σ_row_final, kh_col, kh_row)
        
        i_min1, i_max1, i_min2 = I_min1_map[xi, yi], I_max1_map[xi, yi], I_min2_map[xi, yi]
        i_max2, i_min3 = I_max2_map[xi, yi], I_min3_map[xi, yi]
        
        p_meas = 0.0f0
        if phi_cmd < π
            i_norm = clamp((i_meas - i_min1) / (i_max1 - i_min1 + 1e-6), 0.0f0, 1.0f0)
            p_meas = 2.0f0 * asin(sqrt(i_norm))
        elseif phi_cmd < 2π
            i_norm = clamp((i_meas - i_min2) / (i_max1 - i_min2 + 1e-6), 0.0f0, 1.0f0)
            p_meas = Float32(π) + 2.0f0 * acos(sqrt(i_norm))
        elseif phi_cmd < 3π
            i_norm = clamp((i_meas - i_min2) / (i_max2 - i_min2 + 1e-6), 0.0f0, 1.0f0)
            p_meas = Float32(2π) + 2.0f0 * asin(sqrt(i_norm))
        else
            i_norm = clamp((i_meas - i_min3) / (i_max2 - i_min3 + 1e-6), 0.0f0, 1.0f0)
            p_meas = Float32(3π) + 2.0f0 * acos(sqrt(i_norm))
        end
        push!(valid_phases, p_meas)
    end
    
    if length(valid_phases) > 1
        baseline_rms[v+1]  = (std(valid_phases) / (2.0f0 * π)) * 1000.0f0
        baseline_dphi[v+1] = mean(valid_phases) - phi_cmd
    end
end
println("Baseline verification complete.")

# =========================================================================
# 3. STEP 13.B: AUTOMATED LONGITUDINAL DRIFT LOOP (CALIBRATED REGIONAL LUT)
# =========================================================================
println("\n=== Starting Automated Calibrated Timeline Loop ===")
t_experiment_start = time()

for t_idx in 1:num_time_points
    current_elapsed_target = time_points_mins[t_idx]
    
    if t_idx > 1
        time_elapsed_now = (time() - t_experiment_start) / 60.0
        wait_time_mins = current_elapsed_target - time_elapsed_now
        if wait_time_mins > 0
            @printf("Waiting %.2f minutes to reach next experimental timepoint (%d mins)...\n", wait_time_mins, current_elapsed_target)
            sleep(wait_time_mins * 60.0)
        end
    end
    
    timestamp = Dates.format(Dates.now(), "HH:MM:SS")
    @printf("[%s] Running Calibrated Sweep %d/%d (T = %d mins)\n", timestamp, t_idx, num_time_points, current_elapsed_target)
            
    for p_idx in 1:num_steps
        phi_cmd = phase_axis[p_idx]
        v_lut_idx = clamp(round(Int, p_idx), 1, 256)
        active_frame = copy(phase_dark)
        
        for yi in 1:Ny, xi in 1:Nx
            if pixel_quality_map[xi, yi] == 4 || (pixel_quality_map[xi, yi] == 2 && phi_cmd <= 2π)
                g_val = regional_lut_matrix[xi, yi, v_lut_idx]
                xs, ys = slm_x_range[xi], slm_y_range[yi]
                active_frame[xs, ys] = g_val / 255.0
            end
        end
        
        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)
        cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
        cam_img = Float32.(cam_frame)
        
        valid_phases = Float32[]
        for yi in 1:Ny, xi in 1:Nx
            q = pixel_quality_map[xi, yi]
            (q != 4 && (q != 2 || phi_cmd > 2π)) && continue
            
            i_meas = gaussian_weighted_sample(cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], σ_col_final, σ_row_final, kh_col, kh_row)
            i_min1, i_max1, i_min2 = I_min1_map[xi, yi], I_max1_map[xi, yi], I_min2_map[xi, yi]
            i_max2, i_min3 = I_max2_map[xi, yi], I_min3_map[xi, yi]
            
            p_meas = 0.0f0
            if phi_cmd < π
                i_norm = clamp((i_meas - i_min1) / (i_max1 - i_min1 + 1e-6), 0.0f0, 1.0f0)
                p_meas = 2.0f0 * asin(sqrt(i_norm))
            elseif phi_cmd < 2π
                i_norm = clamp((i_meas - i_min2) / (i_max1 - i_min2 + 1e-6), 0.0f0, 1.0f0)
                p_meas = Float32(π) + 2.0f0 * acos(sqrt(i_norm))
            elseif phi_cmd < 3π && q == 4
                i_norm = clamp((i_meas - i_min2) / (i_max2 - i_min2 + 1e-6), 0.0f0, 1.0f0)
                p_meas = Float32(2π) + 2.0f0 * asin(sqrt(i_norm))
            elseif q == 4
                i_norm = clamp((i_meas - i_min3) / (i_max2 - i_min3 + 1e-6), 0.0f0, 1.0f0)
                p_meas = Float32(3π) + 2.0f0 * acos(sqrt(i_norm))
            end
            push!(valid_phases, p_meas)
        end
        
        if length(valid_phases) > 1
            calib_rms_ts[t_idx, p_idx]  = (std(valid_phases) / (2.0f0 * π)) * 1000.0f0
            calib_dphi_ts[t_idx, p_idx] = mean(valid_phases) - phi_cmd
        end
    end
end

# =========================================================================
# 4. VISUALIZATION GENERATION: FIG 2F & FIG 2H REPRODUCTIONS
# =========================================================================
println("Generating figures...")
line_colors = cgrad(:viridis, num_time_points, categorical=true)

# --- FIGURE 2F: Wavefront RMS Timeline ---
fig_F = Figure(size = (700, 480), font = "Arial")
ax_F = CM.Axis(fig_F[1, 1],
    title = "Figure 2F: Wavefront RMS Error vs Commanded Phase",
    xlabel = "Commanded Phase Retardation (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"])
)
# Plot Uncalibrated Baseline Line
lines!(ax_F, phase_axis, baseline_rms, color = :purple, linewidth = 2.5, linestyle = :dash, label = "Uncalibrated Baseline")
# Plot Calibrated Timeline Elements
for t_idx in 1:num_time_points
    lbl = time_points_mins[t_idx] == 0 ? "Immediate Post-Calib (T=0)" : "$(time_points_mins[t_idx]) mins"
    lines!(ax_F, phase_axis, calib_rms_ts[t_idx, :], color = line_colors[t_idx], linewidth = 2.0, label = lbl)
end
axislegend(ax_F, position = :rt, framevisible = true, bgcolor = :white)
ylims!(ax_F, 0, max(maximum(baseline_rms) * 1.1, 120.0))
display(fig_F)


#save("figure_2F_wavefront_timeline.png", fig_F)

# --- FIGURE 2H: Phase Discrepancy Timeline ---
fig_H = Figure(size = (700, 480), font = "Arial")
ax_H = CM.Axis(fig_H[1, 1],
    title = "Figure 2H: Phase Discrepancy (Measured - Intended)",
    xlabel = "Commanded Phase Retardation (Radians)",
    ylabel = "Delta Phase [rad]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"]),
    yticks = (-π/2:π/4:π/2, ["-¼π", "-⅛π", "0", "⅛π", "¼π"])
)
# Reference line at zero error
hlines!(ax_H, [0.0], color = :gray, linestyle = :dash, linewidth = 1.5)
# Plot Uncalibrated Baseline Line
lines!(ax_H, phase_axis, baseline_dphi, color = :purple, linewidth = 2.5, linestyle = :dash, label = "Uncalibrated Baseline")
# Plot Calibrated Timeline Elements
for t_idx in 1:num_time_points
    lbl = time_points_mins[t_idx] == 0 ? "Immediate Post-Calib (T=0)" : "$(time_points_mins[t_idx]) mins"
    lines!(ax_H, phase_axis, calib_dphi_ts[t_idx, :], color = line_colors[t_idx], linewidth = 2.0, label = lbl)
end
axislegend(ax_H, position = :rt, framevisible = true, bgcolor = :white)
ylims!(ax_H, -π/3, π/3)
display(fig_H)


#save("figure_2H_phase_discrepancy.png", fig_H)

#println("\n[SUCCESS] Figures saved as 'figure_2F_wavefront_timeline.png' and 'figure_2H_phase_discrepancy.png'.")



























#2nd version of step 13
# =========================================================================
# 1. TIMING, PATHS, & GRID CONFIGURATION
# =========================================================================
interval_minutes = 25       # Time between consecutive validation sweeps
total_hours      = 2.5     # Total observation window duration (20 minutes)
num_steps        = 256     # Discrete phase steps per validation run (0 to 4π)

total_minutes = total_hours * 60
time_points_mins = collect(0:interval_minutes:total_minutes)
num_time_points = length(time_points_mins)

Nx, Ny = length(slm_x_range), length(slm_y_range)
kh_col = ceil(Int, 3 * σ_col_final)
kh_row = ceil(Int, 3 * σ_row_final)

# System File Mapping Paths
linear_lut_path = "C:\\Users\\nanolab\\Documents\\MeadowLark\\MeadowLark Lut file\\1024x1024_linearVoltage.lut"
mfg_lut_path    = "C:\\Users\\nanolab\\Documents\\MeadowLark\\MeadowLark Lut file\\slm7831_at633.lut"

# Pre-compute sub-pixel coordinate layout
xc_map_C = zeros(Float64, Nx, Ny)
yc_map_C = zeros(Float64, Nx, Ny)
for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)
        xc_map_C[xi, yi] = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc_map_C[xi, yi] = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    end
end

# Phase Axis and Mask Filters
phase_axis = range(0.0f0, 4f0*π, length=num_steps)
boundary_mask = [all(abs(p - k*π) > 0.1 for k in 0:4) for p in phase_axis]

# Data Tracking Arrays
baseline_rms   = zeros(Float32, num_steps)
baseline_dphi  = zeros(Float32, num_steps)
mfg_rms        = zeros(Float32, num_steps)
mfg_dphi       = zeros(Float32, num_steps)
calib_rms_ts   = zeros(Float32, num_time_points, num_steps)
calib_dphi_ts  = zeros(Float32, num_time_points, num_steps)

# Helper Function: Process a pixel's intensity trace purely by its waveform geometry
function reconstruct_phase_from_sequence(I_trace::Vector{Float32})
    # Smooth trace to filter out high-frequency camera noise for peak finding
    s_trace = copy(I_trace)
    for i in 5:(num_steps-4)
        s_trace[i] = mean(I_trace[(i-4):(i+4)])
    end
    
    # Locate peak and valley indices across the sequence 
    idx_max1 = argmax(s_trace[1:100])
    idx_min2 = 79 + argmin(s_trace[80:180])
    idx_max2 = 149 + argmax(s_trace[150:250])
    # IMPROVEMENT: Dynamically locate the final minimum instead of assuming it's at index 256
    idx_min3 = idx_max2 - 1 + argmin(s_trace[idx_max2:end])
    
    p_reconstructed = zeros(Float32, num_steps)
    
    # Define an isolated overlap blending window size (in steps)
    # This prevents mathematical step discontinuities at turning points
    W = 4 
    
    # Inline helper to evaluate a specific branch phase value safely
    function calc_branch_phase(s_idx, branch_id)
        if branch_id == 1
            i_min, i_max = s_trace[1], s_trace[idx_max1]
            i_norm = clamp((I_trace[s_idx] - i_min) / (i_max - i_min + 1e-6), 0.001f0, 0.999f0)
            return 2.0f0 * asin(sqrt(i_norm))
        elseif branch_id == 2
            i_min, i_max = s_trace[idx_min2], s_trace[idx_max1]
            i_norm = clamp((I_trace[s_idx] - i_min) / (i_max - i_min + 1e-6), 0.001f0, 0.999f0)
            return Float32(π) + 2.0f0 * acos(sqrt(i_norm))
        elseif branch_id == 3
            i_min, i_max = s_trace[idx_min2], s_trace[idx_max2]
            i_norm = clamp((I_trace[s_idx] - i_min) / (i_max - i_min + 1e-6), 0.001f0, 0.999f0)
            return Float32(2π) + 2.0f0 * asin(sqrt(i_norm))
        else
            i_min, i_max = s_trace[idx_min3], s_trace[idx_max2]
            i_norm = clamp((I_trace[s_idx] - i_min) / (i_max - i_min + 1e-6), 0.001f0, 0.999f0)
            return Float32(3π) + 2.0f0 * acos(sqrt(i_norm))
        end
    end

    for s in 1:num_steps
        # Boundary 1: Transition around idx_max1 (π)
        if abs(s - idx_max1) <= W
            w = (s - (idx_max1 - W)) / (2 * W)
            p_reconstructed[s] = (1.0f0 - w) * calc_branch_phase(s, 1) + w * calc_branch_phase(s, 2)
            
        # Boundary 2: Transition around idx_min2 (2π)
        elseif abs(s - idx_min2) <= W
            w = (s - (idx_min2 - W)) / (2 * W)
            p_reconstructed[s] = (1.0f0 - w) * calc_branch_phase(s, 2) + w * calc_branch_phase(s, 3)
            
        # Boundary 3: Transition around idx_max2 (3π)
        elseif abs(s - idx_max2) <= W
            w = (s - (idx_max2 - W)) / (2 * W)
            p_reconstructed[s] = (1.0f0 - w) * calc_branch_phase(s, 3) + w * calc_branch_phase(s, 4)
            
        # Standard Raw Linear Inversion Zones (No Blending Outside Boundary Intersections)
        elseif s < idx_max1
            p_reconstructed[s] = calc_branch_phase(s, 1)
        elseif s < idx_min2
            p_reconstructed[s] = calc_branch_phase(s, 2)
        elseif s < idx_max2
            p_reconstructed[s] = calc_branch_phase(s, 3)
        else
            p_reconstructed[s] = calc_branch_phase(s, 4)
        end
    end
    return p_reconstructed
end

# Helper Function: Process an entire intensity cube into spatial phase map statistics
function analyze_sweep_cube(cube::Array{Float32, 3}, quality_filter::Int)
    rms_profile = zeros(Float32, num_steps)
    dphi_profile = zeros(Float32, num_steps)
    
    spatial_phase_grid = zeros(Float32, Nx, Ny, num_steps)
    
    for yi in 1:Ny, xi in 1:Nx
        q = pixel_quality_map[xi, yi]
        if q == 4 || (quality_filter == 2 && q == 2)
            spatial_phase_grid[xi, yi, :] = reconstruct_phase_from_sequence(cube[xi, yi, :])
        end
    end
    
    for s in 1:num_steps
        valid_phases = Float32[]
        for yi in 1:Ny, xi in 1:Nx
            q = pixel_quality_map[xi, yi]
            if spatial_phase_grid[xi, yi, s] > 0.0f0
                if quality_filter == 4 && q == 4
                    push!(valid_phases, spatial_phase_grid[xi, yi, s])
                elseif quality_filter == 2 && (q == 4 || (q == 2 && phase_axis[s] <= 2π))
                    push!(valid_phases, spatial_phase_grid[xi, yi, s])
                end
            end
        end
        
        if length(valid_phases) > 1
            rms_profile[s] = (std(valid_phases) / (2.0f0 * π)) * 1000.0f0
            dphi_profile[s] = mean(valid_phases) - phase_axis[s]
        end
    end
    return rms_profile, dphi_profile
end


#trying sleep first to see if lowers error in T=0
sleep(3*60)  # Sleep for 3 minutes before starting the automated sequence

# =========================================================================
# 2. RUNTIME LOGISTICS AND AUTOMATED SWEEP EXECUTION
# =========================================================================
println("\n=== Initializing Automated Hardware Verification Sequence ===")
t_experiment_start = time()

# -------------------------------------------------------------------------
# EXECUTION PART A: INITIAL TIMEPOINT RUN (T = 0)
# -------------------------------------------------------------------------
println("\n[T = 0 mins] Executing Calibrated Run 1/$(num_time_points)...")
temp_cube = zeros(Float32, Nx, Ny, num_steps)

for p_idx in 1:num_steps
    phi_cmd = phase_axis[p_idx]
    v_lut_idx = clamp(round(Int, p_idx), 1, 256)
    active_frame = copy(phase_dark)
    
    for yi in 1:Ny, xi in 1:Nx
        if pixel_quality_map[xi, yi] == 4 || (pixel_quality_map[xi, yi] == 2 && phi_cmd <= 2π)
            g_val = regional_lut_matrix[xi, yi, v_lut_idx]
            active_frame[slm_x_range[xi], slm_y_range[yi]] = g_val / 255.0
        end
    end
    
    slm.phase = active_frame
    Meadowlark.writesingleimage(slm)
    cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
    cam_img = Float32.(cam_frame)
    
    for yi in 1:Ny, xi in 1:Nx
        temp_cube[xi, yi, p_idx] = gaussian_weighted_sample(cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], σ_col_final, σ_row_final, kh_col, kh_row)
    end
end
calib_rms_ts[1, :], calib_dphi_ts[1, :] = analyze_sweep_cube(temp_cube, 2)
@printf("   Run 1 Complete. Filtered Average Error: %.2f mλ\n", mean(calib_rms_ts[1, boundary_mask]))

# -------------------------------------------------------------------------
# EXECUTION PART B: INTERLEAVED BASELINE CHARACTERIZATION (INSIDE WAIT WINDOW)
# -------------------------------------------------------------------------
println("\n=== Entering Interleaved Baseline Characterization Window ===")

println("Loading Manufacturer Reference LUT File...")
Meadowlark.loadlut(mfg_lut_path)
fill!(temp_cube, 0.0f0)

for v in 0:255
    active_frame = fill(Float32(v / 255.0), 1024, 1024)
    slm.phase = active_frame
    Meadowlark.writesingleimage(slm)
    cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
    cam_img = Float32.(cam_frame)
    
    for yi in 1:Ny, xi in 1:Nx
        temp_cube[xi, yi, v+1] = gaussian_weighted_sample(cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], σ_col_final, σ_row_final, kh_col, kh_row)
    end
end
mfg_rms, mfg_dphi = analyze_sweep_cube(temp_cube, 4)
println("Manufacturer baseline mapping complete.")

println("Restoring Baseline Linear Voltage Settings...")
Meadowlark.loadlut(linear_lut_path)
fill!(temp_cube, 0.0f0)

for v in 0:255
    active_frame = copy(phase_dark)
    for yi in 1:Ny, xi in 1:Nx
        if pixel_quality_map[xi, yi] == 4
            v_start = min1_map[xi, yi]
            v_end   = min3_map[xi, yi]
            g_val   = v_start + (v / 255.0) * (v_end - v_start)
            active_frame[slm_x_range[xi], slm_y_range[yi]] = clamp(round(Int, g_val), 0, 255) / 255.0
        end
    end
    
    slm.phase = active_frame
    Meadowlark.writesingleimage(slm)
    cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
    cam_img = Float32.(cam_frame)
    
    for yi in 1:Ny, xi in 1:Nx
        temp_cube[xi, yi, v+1] = gaussian_weighted_sample(cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], σ_col_final, σ_row_final, kh_col, kh_row)
    end
end
baseline_rms, baseline_dphi = analyze_sweep_cube(temp_cube, 4)
println("Custom linear uncalibrated baseline mapping complete.")

# -------------------------------------------------------------------------
# EXECUTION PART C: RESUME LONGITUDINAL DRIFT TIMELINE
# -------------------------------------------------------------------------
for t_idx in 2:num_time_points
    current_elapsed_target = time_points_mins[t_idx]
    time_elapsed_now = (time() - t_experiment_start) / 60.0
    wait_time_mins = current_elapsed_target - time_elapsed_now
    
    if wait_time_mins > 0
        @printf("\nWaiting %.2f minutes to reach next scheduled observation checkpoint (%d mins)...\n", wait_time_mins, current_elapsed_target)
        sleep(wait_time_mins * 60.0)
    end
    
    timestamp = Dates.format(Dates.now(), "HH:MM:SS")
    @printf("\n[%s] Executing Calibrated Sweep %d/%d (T = %d mins)...\n", timestamp, t_idx, num_time_points, current_elapsed_target)
    fill!(temp_cube, 0.0f0)
    
    for p_idx in 1:num_steps
        phi_cmd = phase_axis[p_idx]
        v_lut_idx = clamp(round(Int, p_idx), 1, 256)
        active_frame = copy(phase_dark)
        
        for yi in 1:Ny, xi in 1:Nx
            if pixel_quality_map[xi, yi] == 4 || (pixel_quality_map[xi, yi] == 2 && phi_cmd <= 2π)
                g_val = regional_lut_matrix[xi, yi, v_lut_idx]
                active_frame[slm_x_range[xi], slm_y_range[yi]] = g_val / 255.0
            end
        end
        
        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)
        cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
        cam_img = Float32.(cam_frame)
        
        for yi in 1:Ny, xi in 1:Nx
            temp_cube[xi, yi, p_idx] = gaussian_weighted_sample(cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], σ_col_final, σ_row_final, kh_col, kh_row)
        end
    end
    calib_rms_ts[t_idx, :], calib_dphi_ts[t_idx, :] = analyze_sweep_cube(temp_cube, 2)
    @printf("   Run %d Complete. Filtered Average Error: %.2f mλ\n", t_idx, mean(calib_rms_ts[t_idx, boundary_mask]))
end

# =========================================================================
# 3. HIGH-FIDELITY PLOT GENERATION (REPRODUCING FIG 2F & FIG 2H)
# =========================================================================
println("\nGenerating presentation graphics...")
line_colors = cgrad(:viridis, num_time_points, categorical=true)

# --- REPRODUCE FIGURE 2F: Wavefront Tracking Summary ---
fig_F = Figure(size = (720, 500), font = "Arial")
ax_F = CM.Axis(fig_F[1, 1],
    title = "Figure 2F: Spatial Wavefront RMS Error Comparison",
    xlabel = "Commanded Phase Retardation (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"])
)
lines!(ax_F, phase_axis, baseline_rms, color = :purple, linewidth = 2.4, linestyle = :dash, label = "Linear Uncalibrated")
lines!(ax_F, phase_axis, mfg_rms, color = :crimson, linewidth = 2.4, linestyle = :dot, label = "Mfg LUT (633nm file @ 532nm)")
for t_idx in 1:num_time_points
    lbl = time_points_mins[t_idx] == 0 ? "Regional Calib (T=0)" : "Regional Calib ($(time_points_mins[t_idx]) mins)"
    lines!(ax_F, phase_axis, calib_rms_ts[t_idx, :], color = line_colors[t_idx], linewidth = 2.0, label = lbl)
end
axislegend(ax_F, position = :rt, framevisible = true, bgcolor = :white)
ylims!(ax_F, 0, max(maximum(baseline_rms) * 1.1, 150.0))
display(fig_F)

# --- REPRODUCE FIGURE 2H: Phase Discrepancy Map ---
fig_H = Figure(size = (720, 500), font = "Arial")
ax_H = CM.Axis(fig_H[1, 1],
    title = "Figure 2H: Phase Discrepancy (Measured - Intended)",
    xlabel = "Intended Retardation Phase (Radians)",
    ylabel = "Δϕ (Measured - Intended) [rad]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"]),
    yticks = (-π:π/2:π, ["-π", "-½π", "0", "½π", "π"])
)
hlines!(ax_H, [0.0], color = :gray, linestyle = :solid, linewidth = 1.0)
lines!(ax_H, phase_axis, baseline_dphi, color = :purple, linewidth = 2.4, linestyle = :dash, label = "Linear Uncalibrated")
lines!(ax_H, phase_axis, mfg_dphi, color = :crimson, linewidth = 2.4, linestyle = :dot, label = "Mfg LUT")
for t_idx in 1:num_time_points
    lbl = time_points_mins[t_idx] == 0 ? "Regional Calib (T=0)" : "Regional Calib ($(time_points_mins[t_idx]) mins)"
    lines!(ax_H, phase_axis, calib_dphi_ts[t_idx, :], color = line_colors[t_idx], linewidth = 2.0, label = lbl)
end
axislegend(ax_H, position = :rb, framevisible = true, bgcolor = :white)
ylims!(ax_H, -π, π)
display(fig_H)

println("\n[SUCCESS] Figures refreshed using dynamic sequence analysis.")





  
 











#3rd version of step 13
debug_xi = round(Int, Nx/2)
debug_yi = round(Int, Ny/2)

debug_phi = Float32[]
debug_intensity = Float32[]
debug_gray = Int[]
debug_branch = Int[]


# =========================================================================
# 1. CONFIGURATION, TIME TRACKING, & FIXED HARDWARE BOUNDARY MAPS
# =========================================================================
interval_minutes = 7      # Time between consecutive validation sweeps
total_hours      = 0.234     # Total observation window duration
num_steps        = 256     # Discrete phase steps per validation run (0 to 4π)

total_minutes = total_hours * 60  
time_points_mins = collect(0:interval_minutes:total_minutes)
num_time_points = length(time_points_mins)

Nx, Ny = length(slm_x_range), length(slm_y_range)
kh_col = ceil(Int, 3 * σ_col_final)
kh_row = ceil(Int, 3 * σ_row_final)

# Pre-compute floating-point camera coordinate mapping layout
xc_map_C = zeros(Float64, Nx, Ny)
yc_map_C = zeros(Float64, Nx, Ny)
for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)
        xc_map_C[xi, yi] = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc_map_C[xi, yi] = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    end
end

# Cache spatial intensity boundaries for absolute phase branch reconstruction
I_min1_map = zeros(Float32, Nx, Ny)
I_max1_map = zeros(Float32, Nx, Ny)
I_min2_map = zeros(Float32, Nx, Ny)
I_max2_map = zeros(Float32, Nx, Ny)
I_min3_map = zeros(Float32, Nx, Ny)

for yi in 1:Ny, xi in 1:Nx
    if pixel_quality_map[xi, yi] >= 2
        I_min1_map[xi, yi] = intensity_cube[xi, yi, round(Int, min1_map[xi, yi])]
        I_max1_map[xi, yi] = intensity_cube[xi, yi, round(Int, max1_map[xi, yi])]
        I_min2_map[xi, yi] = intensity_cube[xi, yi, round(Int, min2_map[xi, yi])]
    end
    if pixel_quality_map[xi, yi] == 4
        I_max2_map[xi, yi] = intensity_cube[xi, yi, round(Int, max2_map[xi, yi])]
        I_min3_map[xi, yi] = intensity_cube[xi, yi, round(Int, min3_map[xi, yi])]
    end
end

# CRITICAL IMPROVEMENT: Precompute Integer Voltage-Space Phase Branch Boundaries
max1_gray_map = round.(Int, max1_map)
min2_gray_map = round.(Int, min2_map)
max2_gray_map = round.(Int, max2_map)

# Data Storage Matrices
phase_axis     = range(0.0f0, 4f0*π, length=num_steps)
boundary_mask  = [all(abs(p - k*π) > 0.08 for k in 0:4) for p in phase_axis] # Exclude peak-singularity pixels from average metric

baseline_rms   = zeros(Float32, num_steps)
baseline_dphi  = zeros(Float32, num_steps)
calib_rms_ts   = zeros(Float32, num_time_points, num_steps)
calib_dphi_ts  = zeros(Float32, num_time_points, num_steps)

# =========================================================================
# 2. UNIFIED PHASE INVERSION ENGINE (FIXED BOUNDARY INTERFEROMETRY MODEL)
# =========================================================================
"""
    reconstruct_phase_from_intensity(i_meas, g_val, xi, yi)

Inverts the measured pixel intensity back to analytical optical phase. Branch selection is 
governed strictly by the applied integer drive voltage (`g_val`) relative to the fixed spatial 
extrema boundaries established during system calibration.
"""
function reconstruct_phase_from_intensity(i_meas::Float32, g_val::Int, xi::Int, yi::Int)
    # Extract structural voltage thresholds for this local coordinate
    g_max1 = max1_gray_map[xi, yi]
    g_min2 = min2_gray_map[xi, yi]
    g_max2 = max2_gray_map[xi, yi]
    
    # Branch 1: [0, π] -> Increasing Intensity Trace
    if g_val <= g_max1
        i_min, i_max = I_min1_map[xi, yi], I_max1_map[xi, yi]
        i_norm = clamp((i_meas - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
        return 2.0f0 * asin(sqrt(i_norm))
        
    # Branch 2: [π, 2π] -> Decreasing Intensity Trace
    elseif g_val <= g_min2
        i_min, i_max = I_min2_map[xi, yi], I_max1_map[xi, yi]
        i_norm = clamp((i_meas - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
        return Float32(π) + 2.0f0 * acos(sqrt(i_norm))
        
    # Branch 3: [2π, 3π] -> Increasing Intensity Trace
    elseif g_val <= g_max2
        i_min, i_max = I_min2_map[xi, yi], I_max2_map[xi, yi]
        i_norm = clamp((i_meas - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
        return Float32(2π) + 2.0f0 * asin(sqrt(i_norm))
        
    # Branch 4: [3π, 4π] -> Decreasing Intensity Trace
    else
        i_min, i_max = I_min3_map[xi, yi], I_max2_map[xi, yi]
        i_norm = clamp((i_meas - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
        return Float32(3π) + 2.0f0 * acos(sqrt(i_norm))
    end
end

# =========================================================================
# 3. STEP 13.A: EXECUTE UNCALIBRATED LINEAR VOLTAGE BASELINE SWEEP
# =========================================================================
println("\n=== Executing Uncalibrated Baseline Sweep (Linear Voltage Spacing) ===")
for v in 0:255
    phi_cmd = phase_axis[v+1]
    active_frame = copy(phase_dark)
    
    # Assemble raw hardware frame
    for yi in 1:Ny, xi in 1:Nx
        if pixel_quality_map[xi, yi] == 4
            v_start = min1_map[xi, yi]
            v_end   = min3_map[xi, yi]
            g_val   = v_start + (v / 255.0) * (v_end - v_start)
            xs, ys  = slm_x_range[xi], slm_y_range[yi]
            active_frame[xs, ys] = clamp(round(Int, g_val), 0, 255) / 255.0
        end
    end
    
    slm.phase = active_frame
    Meadowlark.writesingleimage(slm)
    cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
    cam_img = Float32.(cam_frame)
    
    valid_phases = Float32[]
    for yi in 1:Ny, xi in 1:Nx
        pixel_quality_map[xi, yi] != 4 && continue
        
        # Extract localized spatial intensity
        i_meas = gaussian_weighted_sample(cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], σ_col_final, σ_row_final, kh_col, kh_row)
        
        # Calculate exactly what applied integer gray byte reached the device backplane
        v_start = min1_map[xi, yi]
        v_end   = min3_map[xi, yi]
        g_applied = clamp(round(Int, v_start + (v / 255.0) * (v_end - v_start)), 0, 255)
        
        # Feed directly to the fixed boundary inversion helper
        #p_meas = reconstruct_phase_from_intensity(i_meas, g_applied, xi, yi)
        g1 = max1_map[xi, yi]
        g2 = min2_map[xi, yi]
        g3 = max2_map[xi, yi]

        if g_applied <= g1
            branch = 1
        elseif g_applied <= g2
            branch = 2
        elseif g_applied <= g3
            branch = 3
        else
            branch = 4
        end

        p_meas = reconstruct_phase_from_intensity(i_meas, g_applied, xi, yi)

        if xi == debug_xi && yi == debug_yi
            push!(debug_phi, p_meas)
            push!(debug_intensity, i_meas)
            push!(debug_gray, g_applied)
            push!(debug_branch, branch)
        end
        push!(valid_phases, p_meas)
    end
    
    if length(valid_phases) > 1
        baseline_rms[v+1]  = (std(valid_phases) / (2.0f0 * π)) * 1000.0f0
        baseline_dphi[v+1] = mean(valid_phases) - phi_cmd
    end
end
println("Baseline verification complete.")

# =========================================================================
# 4. STEP 13.B: AUTOMATED LONGITUDINAL DRIFT LOOP (CALIBRATED REGIONAL LUT)
# =========================================================================
println("\n=== Starting Automated Calibrated Timeline Loop ===")
t_experiment_start = time()

for t_idx in 1:num_time_points
    current_elapsed_target = time_points_mins[t_idx]
    
    if t_idx > 1
        time_elapsed_now = (time() - t_experiment_start) / 60.0
        wait_time_mins = current_elapsed_target - time_elapsed_now
        if wait_time_mins > 0
            @printf("Waiting %.2f minutes to reach next experimental timepoint (%d mins)...\n", wait_time_mins, current_elapsed_target)
            sleep(wait_time_mins * 60.0)
        end
    end
    
    timestamp = Dates.format(Dates.now(), "HH:MM:SS")
    @printf("[%s] Running Calibrated Sweep %d/%d (T = %d mins)\n", timestamp, t_idx, num_time_points, current_elapsed_target)
            
    for p_idx in 1:num_steps
        phi_cmd = phase_axis[p_idx]
        v_lut_idx = clamp(round(Int, p_idx), 1, 256)
        active_frame = copy(phase_dark)
        
        # Generate Calibrated Active Frame Profile
        for yi in 1:Ny, xi in 1:Nx
            q = pixel_quality_map[xi, yi]
            if q == 4 || (q == 2 && phi_cmd <= 2π)
                g_val = regional_lut_matrix[xi, yi, v_lut_idx]
                xs, ys = slm_x_range[xi], slm_y_range[yi]
                active_frame[xs, ys] = g_val / 255.0
            end
        end
        
        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)
        cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
        cam_img = Float32.(cam_frame)
        
        valid_phases = Float32[]
        for yi in 1:Ny, xi in 1:Nx
            q = pixel_quality_map[xi, yi]
            (q != 4 && (q != 2 || phi_cmd > 2π)) && continue
            
            i_meas = gaussian_weighted_sample(cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], σ_col_final, σ_row_final, kh_col, kh_row)
            
            # Fetch the actual discrete integer byte loaded from your regional LUT file
            g_applied = Int(regional_lut_matrix[xi, yi, v_lut_idx])
            
            # Reconstruct using calibration reference thresholds
            #p_meas = reconstruct_phase_from_intensity(i_meas, g_applied, xi, yi)
            g1 = max1_map[xi, yi]
            g2 = min2_map[xi, yi]
            g3 = max2_map[xi, yi]

            if g_applied <= g1
                branch = 1
            elseif g_applied <= g2
                branch = 2
            elseif g_applied <= g3
                branch = 3
            else
                branch = 4
            end

            p_meas = reconstruct_phase_from_intensity(i_meas, g_applied, xi, yi)

            if xi == debug_xi && yi == debug_yi
                push!(debug_phi, p_meas)
                push!(debug_intensity, i_meas)
                push!(debug_gray, g_applied)
                push!(debug_branch, branch)
            end
            push!(valid_phases, p_meas)
        end
        
        if length(valid_phases) > 1
            calib_rms_ts[t_idx, p_idx]  = (std(valid_phases) / (2.0f0 * π)) * 1000.0f0
            calib_dphi_ts[t_idx, p_idx] = mean(valid_phases) - phi_cmd
        end
    end
    @printf("   Run %d Complete. Filtered Average Error: %.2f mλ\n", t_idx, mean(calib_rms_ts[t_idx, boundary_mask]))
end

# =========================================================================
# 5. VISUALIZATION GENERATION: FIG 2F & FIG 2H REPRODUCTIONS
# =========================================================================
println("Generating High-Fidelity Presentation Graphics...")
line_colors = cgrad(:viridis, num_time_points, categorical=true)

# --- FIGURE 2F: Wavefront RMS Timeline ---
fig_F = Figure(size = (720, 500), font = "Arial")
ax_F = CM.Axis(fig_F[1, 1],
    title = "Figure 2F: Spatial Wavefront RMS Error Timeline",
    xlabel = "Commanded Phase Retardation (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"])
)
lines!(ax_F, phase_axis, baseline_rms, color = :purple, linewidth = 2.5, linestyle = :dash, label = "Linear Uncalibrated")
for t_idx in 1:num_time_points
    lbl = time_points_mins[t_idx] == 0 ? "Regional Calib (T=0)" : "Regional Calib ($(time_points_mins[t_idx]) mins)"
    lines!(ax_F, phase_axis, calib_rms_ts[t_idx, :], color = line_colors[t_idx], linewidth = 2.0, label = lbl)
end
axislegend(ax_F, position = :rt, framevisible = true, bgcolor = :white)
ylims!(ax_F, 0, max(maximum(baseline_rms) * 1.1, 120.0))
display(fig_F)

# --- FIGURE 2H: Phase Discrepancy Timeline ---
fig_H = Figure(size = (720, 500), font = "Arial")
ax_H = CM.Axis(fig_H[1, 1],
    title = "Figure 2H: Phase Discrepancy (Measured - Intended)",
    xlabel = "Commanded Phase Retardation (Radians)",
    ylabel = "Delta Phase [rad]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"]),
    yticks = (-π/2:π/4:π/2, ["-¼π", "-⅛π", "0", "⅛π", "¼π"])
)
hlines!(ax_H, [0.0], color = :gray, linestyle = :solid, linewidth = 1.0)
lines!(ax_H, phase_axis, baseline_dphi, color = :purple, linewidth = 2.5, linestyle = :dash, label = "Linear Uncalibrated")
for t_idx in 1:num_time_points
    lbl = time_points_mins[t_idx] == 0 ? "Regional Calib (T=0)" : "Regional Calib ($(time_points_mins[t_idx]) mins)"
    lines!(ax_H, phase_axis, calib_dphi_ts[t_idx, :], color = line_colors[t_idx], linewidth = 2.0, label = lbl)
end
axislegend(ax_H, position = :rt, framevisible = true, bgcolor = :white)
ylims!(ax_H, -π/3, π/3)
display(fig_H)

println("\n[SUCCESS] Figures generated using calibration-anchored voltage branch inversion.")


fig = Figure(size=(1200,900))

############################################################
# Panel 1
############################################################

ax1 = CM.Axis(fig[1,1],
    title="Measured Intensity",
    xlabel="Commanded Phase",
    ylabel="Intensity"
)

lines!(ax1, phase_axis, debug_intensity, linewidth=3)

############################################################
# Panel 2
############################################################

ax2 = CM.Axis(fig[1,2],
    title="Recovered Phase",
    xlabel="Commanded Phase",
    ylabel="Recovered Phase"
)

lines!(ax2, phase_axis, debug_phi,
    color=:blue,
    linewidth=3,
    label="Measured")

lines!(ax2, phase_axis, phase_axis,
    color=:red,
    linestyle=:dash,
    linewidth=2,
    label="Ideal")

axislegend(ax2)

############################################################
# Panel 3
############################################################

ax3 = CM.Axis(fig[2,1],
    title="Applied Grayscale",
    xlabel="Commanded Phase",
    ylabel="Gray Level"
)

lines!(ax3, phase_axis, debug_gray,
    linewidth=3)

hlines!(ax3,
[
    max1_map[debug_xi,debug_yi],
    min2_map[debug_xi,debug_yi],
    max2_map[debug_xi,debug_yi]
],
color=[:red,:green,:blue],
linestyle=:dash)

############################################################
# Panel 4
############################################################

ax4 = CM.Axis(fig[2,2],
    title="Chosen Branch",
    xlabel="Commanded Phase",
    ylabel="Branch",
    yticks=(1:4,["1","2","3","4"])
)

stairs!(ax4,
    phase_axis,
    debug_branch,
    linewidth=3
)

display(fig)
calibration_profile = intensity_cube[debug_xi, debug_yi, :]

fig2 = Figure(size=(800,500))

ax = CM.Axis(fig2[1,1],
    title="Calibration vs Verification Intensity",
    xlabel="Gray Level",
    ylabel="Intensity"
)

lines!(ax,
    0:255,
    calibration_profile,
    linewidth=3,
    label="Calibration")

scatter!(ax,
    debug_gray,
    debug_intensity,
    color=:red,
    markersize=8,
    label="Verification")

axislegend(ax)

display(fig2)










#trying step 13 again... what 4th here?

# =========================================================================
# 1. SETUP PARAMETERS & GRID INITIALIZATION
# =========================================================================
num_steps = 256
phase_axis = range(0.0f0, 4f0*π, length=num_steps)

Nx, Ny = length(slm_x_range), length(slm_y_range)
kh_col = ceil(Int, 3 * σ_col_final)
kh_row = ceil(Int, 3 * σ_row_final)

# Pre-compute floating-point camera coordinate mapping layout to eliminate BoundsErrors
xc_map_C = zeros(Float64, Nx, Ny)
yc_map_C = zeros(Float64, Nx, Ny)
for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)
        xc_map_C[xi, yi] = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc_map_C[xi, yi] = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    end
end

# Allocate results vectors for the 3 target comparison conditions
baseline_mfg_rms   = zeros(Float32, num_steps)
baseline_mfg_dphi  = zeros(Float32, num_steps)

calib_t0_rms       = zeros(Float32, num_steps)
calib_t0_dphi      = zeros(Float32, num_steps)

calib_t45_rms      = zeros(Float32, num_steps)
calib_t45_dphi     = zeros(Float32, num_steps)

# =========================================================================
# 2. HELPER UTILITIES: SMOOTHING & ZERO-CIRCULAR MONOTONIC UNROLLING
# =========================================================================
function smooth_profile_moving_avg(arr::Vector{Float32}, w::Int=5)
    len = length(arr)
    smoothed = copy(arr)
    half_w = div(w, 2)
    for i in (half_w + 1):(len - half_w)
        smoothed[i] = mean(@view arr[(i - half_w):(i + half_w)])
    end
    return smoothed
end

"""
    extract_absolute_phase_from_profile(raw_profile)

Analyzes a raw 256-step intensity profile from a validation sweep, finds its 3 local 
internal turning points from scratch, partitions the data into 4 clean phase quadrants, 
and returns the unrolled measured phase array.
"""
function extract_absolute_phase_from_profile(raw_profile::Vector{Float32})
    num_pts = length(raw_profile)
    measured_phase = fill(NaN32, num_pts)
    
    # 1. Smooth out high-frequency camera noise to protect peak tracking
    smoothed = smooth_profile_moving_avg(raw_profile, 7)
    
    # 2. Track local turning points by scanning derivative zero-crossings
    extrema_indices = Int[]
    for i in 3:(num_pts - 2)
        slope_before = smoothed[i] - smoothed[i-1]
        slope_after  = smoothed[i+1] - smoothed[i]
        if slope_before * slope_after <= 0.0f0 && abs(slope_before) > 1e-5
            # Ensure turning points are physically separated (~1 period/4 ≈ 40 steps)
            if isempty(extrema_indices) || (i - extrema_indices[end] > 35)
                push!(extrema_indices, i)
            end
        end
    end
    
    # Standard 4π stroke expects exactly 3 internal turning points
    if length(extrema_indices) != 3
        return fill(NaN32, num_pts) # Mark as bad pixel if waveform is structurally compromised
    end
    
    k1, k2, k3 = extrema_indices[1], extrema_indices[2], extrema_indices[3]
    segments = [1:k1, (k1+1):k2, (k2+1):k3, (k3+1):num_pts]
    phase_baselines = [0.0f0, Float32(π), Float32(2π), Float32(3π)]
    
    # 3. Process each segment completely independently using Steps 7-10 logic
    for s in 1:4
        idx_range = segments[s]
        length(idx_range) < 2 && continue
        
        seg_intensity = raw_profile[idx_range]
        i_min, i_max = minimum(seg_intensity), maximum(seg_intensity)
        span = i_max - i_min
        span < 5.0f0 && continue # Ignore non-responsive pixels
        
        # Determine if the segment profile is rising or falling
        is_rising = smoothed[idx_range[end]] > smoothed[idx_range[1]]
        
        for (local_idx, global_idx) in enumerate(idx_range)
            norm_i = clamp((raw_profile[global_idx] - i_min) / span, 0.0f0, 1.0f0)
            
            local_phi = is_rising ? 2.0f0 * asin(sqrt(norm_i)) : 2.0f0 * acos(sqrt(norm_i))
            measured_phase[global_idx] = local_phi + phase_baselines[s]
        end
    end
    
    return measured_phase
end

# =========================================================================
# 3. MASTER SWEEP DRIVER (PROJECTS VOLTAGES & SAMPLES INTERFEROMETER)
# =========================================================================
function run_independent_validation_sweep(lut_generation_function)
    # Temporary storage cube for raw intensity readings
    intensity_cube = zeros(Float32, Nx, Ny, num_steps)
    reconstructed_phase_cube = fill(NaN32, Nx, Ny, num_steps)
    
    # Execute hardware hardware step drive
    for p_idx in 1:num_steps
        phi_cmd = phase_axis[p_idx]
        active_frame = copy(phase_dark)
        
        for yi in 1:Ny, xi in 1:Nx
            if pixel_quality_map[xi, yi] == 4
                # Dynamically fetch grayscale value from the targeted LUT routine
                g_val = lut_generation_function(xi, yi, p_idx, phi_cmd)
                active_frame[slm_x_range[xi], slm_y_range[yi]] = clamp(g_val, 0.0f0, 255.0f0) / 255.0
            end
        end
        
        # Project pattern to SLM
        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)
        
        # Capture raw sensor image
        cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
        cam_img = Float32.(cam_frame)
        
        # Sample intensity using localized spatial Gaussian profiles
        for yi in 1:Ny, xi in 1:Nx
            if pixel_quality_map[xi, yi] == 4
                intensity_cube[xi, yi, p_idx] = gaussian_weighted_sample(
                    cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], 
                    σ_col_final, σ_row_final, kh_col, kh_row
                )
            end
        end
    end
    
    # Invert and unroll phase profiles pixel-by-pixel
    for yi in 1:Ny, xi in 1:Nx
        pixel_quality_map[xi, yi] != 4 && continue
        raw_profile = intensity_cube[xi, yi, :]
        reconstructed_phase_cube[xi, yi, :] = extract_absolute_phase_from_profile(raw_profile)
    end
    
    # Calculate aggregation statistics across the spatial wavefront
    rms_out  = zeros(Float32, num_steps)
    dphi_out = zeros(Float32, num_steps)
    
    for p_idx in 1:num_steps
        spatial_slice = filter(!isnan, reconstructed_phase_cube[:, :, p_idx])
        if length(spatial_slice) > 10 # Guarantee statistical relevance
            # Wavefront RMS: spatial deviation converted into millilambda units
            rms_out[p_idx]  = (std(spatial_slice) / (2.0f0 * π)) * 1000.0f0
            # Phase Discrepancy: mean spatial deviation from commanded target phase
            dphi_out[p_idx] = mean(spatial_slice) - phase_axis[p_idx]
        end
    end
    
    return rms_out, dphi_out
end

# =========================================================================
# 4. EXECUTION TIMELINE RUNNER
# =========================================================================

# --- Condition 1: Manufacturer Provided Linear Reference LUT ---
println("\n>>> Running Manufacturer Default Linear Reference Sweep...")
mfg_lut_func(xi, yi, p_idx, phi_cmd) = (phi_cmd / (4.0f0 * π)) * 255.0f0
baseline_mfg_rms, baseline_mfg_dphi = run_independent_validation_sweep(mfg_lut_func)

# --- Condition 2: Custom Regional LUT (Immediate Post-Calibration) ---
println("\n>>> Running Custom Regional LUT Sweep [T = 0 Minutes]...")
t0_lut_func(xi, yi, p_idx, phi_cmd) = regional_lut_matrix[xi, yi, clamp(p_idx, 1, 256)]
calib_t0_rms, calib_t0_dphi = run_independent_validation_sweep(t0_lut_func)

# --- Condition 3: Custom Regional LUT (Longitudinal Drift Delay) ---
println("\n>>> Entering 45-Minute Environmental Drift Hold Phase...")
sleep(10 * 60.0) # Wait 45 minutes to benchmark regional stability 

println("\n>>> Running Custom Regional LUT Sweep [T = 45 Minutes]...")
calib_t45_rms, calib_t45_dphi = run_independent_validation_sweep(t0_lut_func)

# =========================================================================
# 5. DIAGNOSTIC PLOT GENERATION (REPLICATING FIGURES 2F & 2H)
# =========================================================================
println("\nGenerating final paper-grade figure metrics...")

# --- FIGURE 2F: Wavefront RMS Error Comparison ---
fig_2F = Figure(size = (750, 520), font = "Arial")
ax_2F = CM.Axis(fig_2F[1, 1],
    title = "Figure 2F: Wavefront Spatial Aberration Stability",
    xlabel = "Intended Phase Retardation (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"])
)
lines!(ax_2F, phase_axis, baseline_mfg_rms, color = :black, linewidth = 2.0, linestyle = :dash, label = "Manufacturer LUT")
lines!(ax_2F, phase_axis, calib_t0_rms,       color = :deepskyblue, linewidth = 2.5, label = "Custom LUT (T = 0 min)")
lines!(ax_2F, phase_axis, calib_t45_rms,      color = :crimson, linewidth = 2.0, label = "Custom LUT (T = 45 min)")
axislegend(ax_2F, position = :rt, framevisible = true, bgcolor = :white)
ylims!(ax_2F, 0, max(maximum(calib_t45_rms)*1.4, 35.0))
display(fig_2F)

# --- FIGURE 2H: Spatial Average Phase Discrepancy ---
fig_2H = Figure(size = (750, 520), font = "Arial")
ax_2H = CM.Axis(fig_2H[1, 1],
    title = "Figure 2H: Phase Discrepancy Trace",
    xlabel = "Intended Phase Retardation (Radians)",
    ylabel = "Measured - Intended Phase Δϕ [rad]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"]),
    yticks = (-π/2:π/4:π/2, ["-½π", "-¼π", "0", "¼π", "½π"])
)
hlines!(ax_2H, [0.0], color = :gray, linestyle = :solid, linewidth = 1.0)
lines!(ax_2H, phase_axis, baseline_mfg_dphi, color = :black, linewidth = 2.0, linestyle = :dash, label = "Manufacturer LUT")
lines!(ax_2H, phase_axis, calib_t0_dphi,       color = :deepskyblue, linewidth = 2.5, label = "Custom LUT (T = 0 min)")
lines!(ax_2H, phase_axis, calib_t45_dphi,      color = :crimson, linewidth = 2.0, label = "Custom LUT (T = 45 min)")
axislegend(ax_2H, position = :rt, framevisible = true, bgcolor = :white)
ylims!(ax_2H, -π/2, π/2)
display(fig_2H)

println("Execution cycle complete. Plots rendered successfully.")



















#5th version, hopefuly 5th is the charm lol
 
# ── Timing configuration ───────────────────────────────────────────────────────
const INTERVAL_MINUTES  = 7       # gap between calibrated sweeps
const TOTAL_MINUTES     = 28      # total longitudinal window
const NUM_STEPS         = 256     # phase steps per sweep (0 → 4π)
const PHASE_AXIS        = Float32.(range(0.0, 4π, length=NUM_STEPS))
 
time_points_mins = collect(0:INTERVAL_MINUTES:TOTAL_MINUTES)
num_time_points  = length(time_points_mins)
 
# ── Spatial dimensions (from upstream calibration) ─────────────────────────────
Nx, Ny   = length(slm_x_range), length(slm_y_range)
kh_col_v = ceil(Int, 3 * σ_col_final)
kh_row_v = ceil(Int, 3 * σ_row_final)
 
# ── Pre-compute affine camera coordinate map (once) ───────────────────────────
xc_map_V = zeros(Float64, Nx, Ny)
yc_map_V = zeros(Float64, Nx, Ny)
for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)
        xc_map_V[xi, yi] =
            affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc_map_V[xi, yi] =
            affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    end
end
 
# ── Boundary exclusion mask for scalar RMS summary ────────────────────────────
# Steps within BOUNDARY_HALF steps of kπ are numerically ill-conditioned
# (arcsin/arccos derivative → ∞ at endpoints). They are measured and stored
# but excluded from the scalar mean RMS printed in the console summary.
const BOUNDARY_HALF = 5
boundary_step_width = (4f0 * Float32(π)) / NUM_STEPS
boundary_include = [
    all(abs(PHASE_AXIS[p] - k*Float32(π)) > BOUNDARY_HALF * boundary_step_width
        for k in 0:4)
    for p in 1:NUM_STEPS
]
println("Steps included in scalar RMS summary: $(sum(boundary_include)) / $NUM_STEPS")
 
# ==============================================================================
# Helper: capture one camera frame
# ==============================================================================
function capture_frame()
    return Float32.(try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end)
end
 
# ==============================================================================
# Helper: smooth a 1-D Float32 profile with a simple moving average
# (identical to the smooth_profile used in Step 8)
# ==============================================================================
function smooth_profile_v(v::AbstractVector{Float32}; half::Int=2)
    n   = length(v)
    out = similar(v)
    for i in 1:n
        out[i] = mean(v[max(1,i-half):min(n,i+half)])
    end
    return out
end
 
# ==============================================================================
# Helper: locate extrema in a 1-D profile (mirrors Step 8 logic exactly)
#
# Returns a NamedTuple:
#   min1_v, max1_v, min2_v  :: Float64  (always present if quality ≥ 2)
#   max2_v, min3_v          :: Float64  (present if quality == 4, else 0.0)
#   quality                 :: Int      (4, 2, or 0)
#
# search_lo / search_hi are 1-based indices into the profile that bound
# the search window. Pass the full range (1:NUM_STEPS) for a general sweep.
# ==============================================================================
function fit_parabolic_vertex_v(idx::Int, profile::AbstractVector)
    y_m = profile[idx-1]
    y_0 = profile[idx]
    y_p = profile[idx+1]
    denom = 2*(y_m + y_p - 2*y_0)
    abs(denom) < 1e-9 && return Float64(idx)
    return idx + (y_m - y_p) / denom
end
 
function find_extrema_v(profile::AbstractVector{Float32};
                         search_lo::Int=1, search_hi::Int=length(profile),
                         min_start_idx::Int=1)   # first minimum must be ≥ this index
    sm = smooth_profile_v(profile)
 
    local_mins  = Int[]
    local_maxes = Int[]
    for i in (search_lo+1):(search_hi-1)
        if sm[i] < sm[i-1] && sm[i] <= sm[i+1]
            push!(local_mins, i)
        elseif sm[i] > sm[i-1] && sm[i] >= sm[i+1]
            push!(local_maxes, i)
        end
    end
 
    result = (min1_v=0.0, max1_v=0.0, min2_v=0.0,
               max2_v=0.0, min3_v=0.0, quality=0)
 
    try
        # Min1
        fi = findfirst(m -> m >= min_start_idx, local_mins)
        isnothing(fi) && return result
        m1 = local_mins[fi]
 
        # Max1
        fi = findfirst(x -> x > m1, local_maxes)
        isnothing(fi) && return result
        x1 = local_maxes[fi]
 
        # Min2
        fi = findfirst(m -> m > x1, local_mins)
        isnothing(fi) && return result
        m2 = local_mins[fi]
 
        min1_v = fit_parabolic_vertex_v(m1, sm)
        max1_v = fit_parabolic_vertex_v(x1, sm)
        min2_v = fit_parabolic_vertex_v(m2, sm)
 
        # Try for 4π (Max2 + Min3)
        fi_x2 = findfirst(x -> x > m2, local_maxes)
        fi_m3 = findfirst(m -> m > m2 + 10, local_mins)
 
        if !isnothing(fi_x2) && !isnothing(fi_m3)
            x2 = local_maxes[fi_x2]
            m3 = local_mins[fi_m3]
            if m1 < x1 < m2 < x2 < m3
                max2_v = fit_parabolic_vertex_v(x2, sm)
                min3_v = fit_parabolic_vertex_v(m3, sm)
                return (min1_v=min1_v, max1_v=max1_v, min2_v=min2_v,
                        max2_v=max2_v, min3_v=min3_v, quality=4)
            end
        end
 
        return (min1_v=min1_v, max1_v=max1_v, min2_v=min2_v,
                max2_v=0.0, min3_v=0.0, quality=2)
 
    catch
        return result
    end
end
 
# ==============================================================================
# Helper: recover phase for one pixel from its full 256-step intensity trace
#         and its fresh extrema.
#
# This mirrors Step 10: intensity → [0,1] per segment → arcsin/arccos.
# Branch boundaries come ONLY from the fresh extrema, not from calibration maps.
# Returns a Float32 vector of length NUM_STEPS (NaN where reconstruction fails).
# ==============================================================================
function recover_phase_from_trace(trace::AbstractVector{Float32}, ext)
    phi = fill(NaN32, NUM_STEPS)
    ext.quality == 0 && return phi
 
    # Segment boundaries in index space (1-based)
    m1 = round(Int, ext.min1_v)
    x1 = round(Int, ext.max1_v)
    m2 = round(Int, ext.min2_v)
 
    # Local extremum intensities from the trace itself
    I_min1 = trace[m1]
    I_max1 = trace[x1]
    I_min2 = trace[m2]
 
    # Segment 1: indices 1 → x1  (phase 0 → π, rising)
    for i in 1:x1
        denom = I_max1 - I_min1
        abs(denom) < 1f0 && continue
        yn = clamp((trace[i] - I_min1) / denom, 0.02f0, 0.98f0)
        phi[i] = 2f0 * asin(sqrt(yn))
    end
 
    # Segment 2: indices x1 → m2  (phase π → 2π, falling)
    for i in x1:m2
        denom = I_max1 - I_min2
        abs(denom) < 1f0 && continue
        yn = clamp((trace[i] - I_min2) / denom, 0.02f0, 0.98f0)
        phi[i] = Float32(π) + 2f0 * acos(sqrt(yn))
    end
 
    ext.quality == 2 && return phi
 
    x2 = round(Int, ext.max2_v)
    m3 = round(Int, ext.min3_v)
    I_max2 = trace[x2]
    I_min3 = trace[m3]
 
    # Segment 3: indices m2 → x2  (phase 2π → 3π, rising)
    for i in m2:x2
        denom = I_max2 - I_min2
        abs(denom) < 1f0 && continue
        yn = clamp((trace[i] - I_min2) / denom, 0.02f0, 0.98f0)
        phi[i] = Float32(2π) + 2f0 * asin(sqrt(yn))
    end
 
    # Segment 4: indices x2 → NUM_STEPS  (phase 3π → 4π, falling)
    for i in x2:NUM_STEPS
        denom = I_max2 - I_min3
        abs(denom) < 1f0 && continue
        yn = clamp((trace[i] - I_min3) / denom, 0.02f0, 0.98f0)
        phi[i] = Float32(3π) + 2f0 * acos(sqrt(yn))
    end
 
    return phi
end
 
# ==============================================================================
# Core: run one complete validation sweep and return (rms, dphi) per step
#
# frame_builder(step_idx::Int) → Matrix{Float32}  SLM phase matrix to write
#   step_idx runs 1:NUM_STEPS
#
# The function:
#   1. Captures a 256-step intensity cube  (mirrors Step 7 / Method C)
#   2. Finds fresh extrema per pixel       (mirrors Step 8)
#   3. Recovers phase per pixel            (mirrors Step 10)
#   4. Computes RMS (spatial std) and mean Δφ per commanded phase step
# ==============================================================================
function run_validation_sweep(frame_builder::Function; label::String="")
 
    # ── 1. Capture intensity cube ──────────────────────────────────────────────
    val_cube = zeros(Float32, Nx, Ny, NUM_STEPS)
 
    for step in 1:NUM_STEPS
        slm.phase = frame_builder(step)
        Meadowlark.writesingleimage(slm)
 
        cam_img = capture_frame()
 
        for xi in 1:Nx, yi in 1:Ny
            val_cube[xi, yi, step] = gaussian_weighted_sample(
                cam_img,
                xc_map_V[xi, yi], yc_map_V[xi, yi],
                σ_col_final, σ_row_final,
                kh_col_v, kh_row_v)
        end
 
        if step % 64 == 0
            @printf("  [%s] captured step %3d/%d\n", label, step, NUM_STEPS)
        end
    end
 
    # ── 2. Fresh extrema per pixel (Step 8 mirrored) ──────────────────────────
    # For a calibrated sweep the intensity oscillations may not begin at step 1
    # (since the LUT already linearises phase). We search the full range and
    # let find_extrema_v locate wherever the fringes actually appear.
    ext_map = Array{NamedTuple}(undef, Nx, Ny)
    for xi in 1:Nx, yi in 1:Ny
        trace = val_cube[xi, yi, :]
        ext_map[xi, yi] = find_extrema_v(trace;
                                           search_lo=2,
                                           search_hi=NUM_STEPS-1,
                                           min_start_idx=1)
    end
 
    q4 = count(e -> e.quality == 4, ext_map)
    q2 = count(e -> e.quality == 2, ext_map)
    q0 = count(e -> e.quality == 0, ext_map)
    @printf("  [%s] extrema: 4π=%d  2π=%d  failed=%d\n", label, q4, q2, q0)
 
    # ── 3. Phase recovery per pixel (Step 10 mirrored) ────────────────────────
    # phi_cube[xi, yi, step] = recovered phase in radians (NaN if failed)
    phi_cube = Array{Float32}(undef, Nx, Ny, NUM_STEPS)
    for xi in 1:Nx, yi in 1:Ny
        phi_cube[xi, yi, :] = recover_phase_from_trace(
            val_cube[xi, yi, :], ext_map[xi, yi])
    end
 
    # ── 4. Per-step statistics ─────────────────────────────────────────────────
    rms_arr  = zeros(Float32, NUM_STEPS)
    dphi_arr = zeros(Float32, NUM_STEPS)
 
    for step in 1:NUM_STEPS
        phi_cmd = PHASE_AXIS[step]
 
        valid = Float32[]
        for xi in 1:Nx, yi in 1:Ny
            v = phi_cube[xi, yi, step]
            isnan(v) && continue
            # For 2π-only pixels, skip steps that would require 4π reconstruction
            ext_map[xi, yi].quality == 2 && phi_cmd > 2f0*Float32(π) && continue
            push!(valid, v)
        end
 
        if length(valid) > 1
            # RMS = spatial std / (2π) × 1000, in units of mλ
            rms_arr[step]  = (std(valid)  / (2f0 * Float32(π))) * 1000f0
            dphi_arr[step] = mean(valid) - phi_cmd
        end
    end
 
    return rms_arr, dphi_arr
end
 
# ==============================================================================
# Frame builders — one per LUT condition
# ==============================================================================
 
# (A) Calibrated: look up per-pixel grey value from regional_lut_matrix
function frame_calibrated(step::Int)
    frame = copy(phase_dark)
    phi_cmd = PHASE_AXIS[step]
    for xi in 1:Nx, yi in 1:Ny
        q = pixel_quality_map[xi, yi]
        (q != 4 && !(q == 2 && phi_cmd <= 2f0*Float32(π))) && continue
        g = regional_lut_matrix[xi, yi, step]   # UInt8
        frame[slm_x_range[xi], slm_y_range[yi]] = Float32(g) / 255f0
    end
    return frame
end
 
# (B) Linear uncalibrated: linearly interpolate grey value between per-pixel
#     min1 and min3 voltage boundaries (uses calibration extrema only to define
#     the voltage *range*, not to reconstruct phase during validation)
function frame_linear(step::Int)
    frame = copy(phase_dark)
    t = (step - 1) / (NUM_STEPS - 1)    # 0.0 → 1.0
    for xi in 1:Nx, yi in 1:Ny
        pixel_quality_map[xi, yi] != 4 && continue
        v_lo = min1_map[xi, yi]          # Float64, sub-pixel voltage index
        v_hi = min3_map[xi, yi]
        g    = clamp(round(Int, v_lo + t * (v_hi - v_lo)), 0, 255)
        frame[slm_x_range[xi], slm_y_range[yi]] = Float32(g) / 255f0
    end
    return frame
end
 
# (C) Manufacturer LUT: simple uniform grey ramp 0→255.
#     The 633nm LUT is loaded before this sweep and reloaded after.
function frame_manufacturer(step::Int)
    g_norm = Float32(step - 1) / Float32(NUM_STEPS - 1)
    frame  = fill(g_norm, 1024, 1024)
    return frame
end
 
# ==============================================================================
# Storage
# ==============================================================================
calib_rms_ts  = zeros(Float32, num_time_points, NUM_STEPS)
calib_dphi_ts = zeros(Float32, num_time_points, NUM_STEPS)
linear_rms    = zeros(Float32, NUM_STEPS)
linear_dphi   = zeros(Float32, NUM_STEPS)
mfr_rms       = zeros(Float32, NUM_STEPS)
mfr_dphi      = zeros(Float32, NUM_STEPS)
 
# ==============================================================================
# EXECUTION
# ==============================================================================
 
t_exp_start = time()
 
# ── 1. Calibrated T=0 ─────────────────────────────────────────────────────────
println("\n=== Calibrated sweep 1/$(num_time_points)  (T = 0 min) ===")
calib_rms_ts[1,:], calib_dphi_ts[1,:] =
    run_validation_sweep(frame_calibrated; label="calib T=0")
@printf("  Mean RMS (boundary-excluded) = %.1f mλ\n",
        mean(calib_rms_ts[1, boundary_include]))
 
# ── 2. Interleaved baselines (run during the first wait window) ───────────────
 
# 2a. Manufacturer LUT sweep
println("\n=== Manufacturer 633nm LUT sweep ===")
Meadowlark.loadlut(MFR_LUT_PATH)
println("  slm7831_at633.lut loaded.")
mfr_rms, mfr_dphi =
    run_validation_sweep(frame_manufacturer; label="mfr LUT")
Meadowlark.loadlut(LINEAR_LUT_PATH)
println("  Linear LUT restored.")
@printf("  Manufacturer LUT mean RMS = %.1f mλ\n",
        mean(mfr_rms[boundary_include]))
 
# 2b. Linear voltage baseline sweep (under linear LUT)
println("\n=== Linear voltage baseline sweep ===")
linear_rms, linear_dphi =
    run_validation_sweep(frame_linear; label="linear")
@printf("  Linear baseline mean RMS = %.1f mλ\n",
        mean(linear_rms[boundary_include]))
 
# ── 3. Remaining calibrated sweeps with scheduled delays ──────────────────────
for t_idx in 2:num_time_points
    target_min  = time_points_mins[t_idx]
    elapsed_min = (time() - t_exp_start) / 60.0
    wait_min    = target_min - elapsed_min
 
    if wait_min > 0
        @printf("\nWaiting %.1f min → T=%d min checkpoint...\n",
                wait_min, target_min)
        sleep(wait_min * 60.0)
    end
 
    ts = Dates.format(Dates.now(), "HH:MM:SS")
    @printf("\n=== [%s] Calibrated sweep %d/%d  (T = %d min) ===\n",
            ts, t_idx, num_time_points, target_min)
 
    calib_rms_ts[t_idx,:], calib_dphi_ts[t_idx,:] =
        run_validation_sweep(frame_calibrated; label="calib T=$(target_min)")
    @printf("  Mean RMS = %.1f mλ\n",
            mean(calib_rms_ts[t_idx, boundary_include]))
end
 
println("\n=== All sweeps complete. Generating figures. ===")
 
# ==============================================================================
# Console summary table
# ==============================================================================
println("\n─── Scalar RMS summary (boundary steps excluded) ───")
println("  Condition                  │ Mean RMS (mλ) │ Max RMS (mλ)")
@printf("  %-26s│ %13.1f │ %12.1f\n",
        "Manufacturer 633nm LUT",
        mean(mfr_rms[boundary_include]),
        maximum(mfr_rms[boundary_include]))
@printf("  %-26s│ %13.1f │ %12.1f\n",
        "Linear voltage baseline",
        mean(linear_rms[boundary_include]),
        maximum(linear_rms[boundary_include]))
for t_idx in 1:num_time_points
    lbl = "Calibrated T=$(time_points_mins[t_idx]) min"
    @printf("  %-26s│ %13.1f │ %12.1f\n",
            lbl,
            mean(calib_rms_ts[t_idx, boundary_include]),
            maximum(calib_rms_ts[t_idx, boundary_include]))
end
 
# ==============================================================================
# Fig 2.F — Wavefront RMS error vs commanded phase
# ==============================================================================
line_colors = cgrad(:viridis, num_time_points, categorical=true)
pa = collect(PHASE_AXIS)   # plain Vector for plotting
 
fig_F = Figure(size=(800, 500))
ax_F  = CM.Axis(fig_F[1,1],
    title  = "Fig 2.F  Wavefront RMS error vs commanded phase",
    xlabel = "Commanded phase retardation (rad)",
    ylabel = "W_rms (mλ)",
    xticks = (Float32.(0:π:4π), ["0", "π", "2π", "3π", "4π"]))
 
# Shade excluded boundary regions
for k in 0:4
    phi_k = k * Float32(π)
    vspan!(ax_F,
           phi_k - BOUNDARY_HALF * boundary_step_width,
           phi_k + BOUNDARY_HALF * boundary_step_width,
           color=(:gray, 0.12))
end
 
lines!(ax_F, pa, mfr_rms,
       color=:purple, linewidth=2.5, linestyle=:dash,
       label="Manufacturer 633nm LUT")
lines!(ax_F, pa, linear_rms,
       color=:orange, linewidth=2.5, linestyle=:dot,
       label="Linear voltage (uncalibrated)")
 
for t_idx in 1:num_time_points
    t_min = time_points_mins[t_idx]
    lbl   = t_min == 0 ? "Calibrated  T = 0" : "Calibrated  T = $(t_min) min"
    lines!(ax_F, pa, calib_rms_ts[t_idx,:],
           color=line_colors[t_idx], linewidth=2.0, label=lbl)
end
 
axislegend(ax_F, position=:rt, framevisible=true)
ylims!(ax_F, 0,
       max(maximum(mfr_rms) * 1.1,
           maximum(linear_rms) * 1.1,
           120.0f0))
display(fig_F)
 
# ==============================================================================
# Fig 2.H — Phase discrepancy (mean measured − commanded)
# ==============================================================================
fig_H = Figure(size=(800, 500))
ax_H  = CM.Axis(fig_H[1,1],
    title  = "Fig 2.H  Phase discrepancy: mean measured − commanded",
    xlabel = "Commanded phase retardation (rad)",
    ylabel = "Δφ  (rad)",
    xticks = (Float32.(0:π:4π), ["0", "π", "2π", "3π", "4π"]),
    yticks = (Float32.([-π/2, -π/4, 0, π/4, π/2]),
              ["-π/2", "-π/4", "0", "π/4", "π/2"]))
 
for k in 0:4
    phi_k = k * Float32(π)
    vspan!(ax_H,
           phi_k - BOUNDARY_HALF * boundary_step_width,
           phi_k + BOUNDARY_HALF * boundary_step_width,
           color=(:gray, 0.12))
end
 
hlines!(ax_H, [0.0f0], color=:gray, linestyle=:dash, linewidth=1.0)
 
lines!(ax_H, pa, mfr_dphi,
       color=:purple, linewidth=2.5, linestyle=:dash,
       label="Manufacturer 633nm LUT")
lines!(ax_H, pa, linear_dphi,
       color=:orange, linewidth=2.5, linestyle=:dot,
       label="Linear voltage (uncalibrated)")
 
for t_idx in 1:num_time_points
    t_min = time_points_mins[t_idx]
    lbl   = t_min == 0 ? "Calibrated  T = 0" : "Calibrated  T = $(t_min) min"
    lines!(ax_H, pa, calib_dphi_ts[t_idx,:],
           color=line_colors[t_idx], linewidth=2.0, label=lbl)
end
 
axislegend(ax_H, position=:rt, framevisible=true)
ylims!(ax_H, -Float32(π)/3, Float32(π)/3)
display(fig_H)
 
# ==============================================================================
# Optional single-pixel diagnostic: shows exactly what the validation pipeline
# sees for one representative pixel — useful for debugging branch errors.
# ==============================================================================
diag_xi = Nx ÷ 2
diag_yi = Ny ÷ 2
while pixel_quality_map[diag_xi, diag_yi] != 4 && diag_xi < Nx
    diag_xi += 1
end
 
# Re-run one calibrated sweep just for this pixel to get its trace
# (we already have it from the last calibrated sweep's val_cube, but
# that variable is local to run_validation_sweep. Easiest to pull from
# the last calib run by re-using the frame builder on a tiny fresh capture.)
println("\nGenerating single-pixel diagnostic for SLM pixel " *
        "($(slm_x_range[diag_xi]), $(slm_y_range[diag_yi]))...")
 
diag_trace = zeros(Float32, NUM_STEPS)
for step in 1:NUM_STEPS
    slm.phase = frame_calibrated(step)
    Meadowlark.writesingleimage(slm)
    cam_img = capture_frame()
    diag_trace[step] = gaussian_weighted_sample(
        cam_img,
        xc_map_V[diag_xi, diag_yi], yc_map_V[diag_xi, diag_yi],
        σ_col_final, σ_row_final, kh_col_v, kh_row_v)
    step % 64 == 0 && println("  diagnostic step $step/$(NUM_STEPS)")
end
 
diag_ext = find_extrema_v(diag_trace; search_lo=2, search_hi=NUM_STEPS-1)
diag_phi = recover_phase_from_trace(diag_trace, diag_ext)
 
fig_diag = Figure(size=(1100, 450))
 
ax_d1 = CM.Axis(fig_diag[1,1],
    title  = "Validation intensity trace — pixel $(slm_x_range[diag_xi]),$(slm_y_range[diag_yi])",
    xlabel = "Commanded phase step",
    ylabel = "Gaussian-weighted intensity (ADU)")
lines!(ax_d1, pa, diag_trace, color=:steelblue, linewidth=2)
for (name, val, col) in [
        ("min1", diag_ext.min1_v, :red),
        ("max1", diag_ext.max1_v, :green),
        ("min2", diag_ext.min2_v, :red),
        ("max2", diag_ext.max2_v, :green),
        ("min3", diag_ext.min3_v, :red)]
    val == 0.0 && continue
    # convert step-index to phase value for x-axis
    phi_at_idx = PHASE_AXIS[clamp(round(Int, val), 1, NUM_STEPS)]
    vlines!(ax_d1, [phi_at_idx], color=col, linestyle=:dash, linewidth=1.5)
end
 
ax_d2 = CM.Axis(fig_diag[1,2],
    title  = "Recovered vs commanded phase",
    xlabel = "Commanded phase (rad)",
    ylabel = "Recovered phase (rad)",
    xticks = (Float32.(0:π:4π), ["0","π","2π","3π","4π"]),
    yticks = (Float32.(0:π:4π), ["0","π","2π","3π","4π"]))
lines!(ax_d2, pa, diag_phi, color=:crimson, linewidth=2, label="Measured")
lines!(ax_d2, pa, pa,       color=:black,  linewidth=1,
       linestyle=:dash, label="Ideal")
axislegend(ax_d2, position=:lt)
 
display(fig_diag)
println("\nStep 13 complete.")











#6th code, gemini, pls work!!!
# 1. TIMING, GRID SELECTION, & GLOBAL LUT PRE-COMPUTATION
# =========================================================================
num_steps = 256
phase_axis = range(0.0f0, 4f0*π, length=num_steps)
# Mask out turning point singularities (±0.15 rad around multiples of π) for accurate baseline reporting
singularity_mask = [abs(p - π) > 0.15 && abs(p - 2π) > 0.15 && abs(p - 3π) > 0.15 for p in phase_axis]

Nx, Ny = length(slm_x_range), length(slm_y_range)
kh_col = ceil(Int, 3 * σ_col_final)
kh_row = ceil(Int, 3 * σ_row_final)

# Pre-compute floating-point camera coordinate mapping layout to eliminate BoundsErrors
xc_map_C = zeros(Float64, Nx, Ny)
yc_map_C = zeros(Float64, Nx, Ny)
for (xi, xs) in enumerate(slm_x_range)
    for (yi, ys) in enumerate(slm_y_range)
        xc_map_C[xi, yi] = affine_matrix[1,1,1]*xs + affine_matrix[1,2,1]*ys + affine_matrix[1,3,1]
        yc_map_C[xi, yi] = affine_matrix[1,1,2]*xs + affine_matrix[1,2,2]*ys + affine_matrix[1,3,2]
    end
end

# Compute the Global Calibrated LUT (The spatial average of your custom regional matrix)
global_calib_lut = zeros(Float32, num_steps)
for p in 1:num_steps
    valid_pixels = filter(v -> v > 0.0f0 && v <= 255.0f0, regional_lut_matrix[:, :, p])
    global_calib_lut[p] = isempty(valid_pixels) ? 0.0f0 : mean(valid_pixels)
end

# Allocate results vectors for all 4 distinct verification configurations
mfg_rms, mfg_dphi           = zeros(Float32, num_steps), zeros(Float32, num_steps)
lin_v_rms, lin_v_dphi       = zeros(Float32, num_steps), zeros(Float32, num_steps)
global_rms, global_dphi     = zeros(Float32, num_steps), zeros(Float32, num_steps)
regional_t0_rms, regional_t0_dphi   = zeros(Float32, num_steps), zeros(Float32, num_steps)
regional_t45_rms, regional_t45_dphi = zeros(Float32, num_steps), zeros(Float32, num_steps)

# Matrix to hold validation quality map for Figure 2G
validation_quality_map = zeros(Int, Nx, Ny)

# =========================================================================
# 2. CORE MATHEMATICAL UNROLLING ENGINE (FROM SCRATCH, NO CIRCULAR LOGIC)
# =========================================================================
function smooth_profile_validation(arr::Vector{Float32}, w::Int=2)
    len = length(arr)
    smoothed = copy(arr)
    for i in (w + 1):(len - w)
        smoothed[i] = mean(@view arr[(i - w):(i + w)])
    end
    return smoothed
end

function fit_validation_vertex(center_idx::Int, smoothed_profile::Vector{Float32})
    len = length(smoothed_profile)
    if center_idx <= 1 || center_idx >= len
        return Float32(center_idx)
    end
    y1 = smoothed_profile[center_idx - 1]
    y2 = smoothed_profile[center_idx]
    y3 = smoothed_profile[center_idx + 1]
    denom = 2.0f0 * (y1 - 2.0f0 * y2 + y3)
    if abs(denom) < 1e-5
        return Float32(center_idx)
    end
    num = y1 - y3
    vertex = Float32(center_idx) + (num / denom)
    return clamp(vertex, 1.0f0, Float32(len))
end

"""
    extract_absolute_phase_from_fresh_sweep(raw_profile)
Inverts a completely fresh intensity sweep into an absolute phase vector by tracking local 
extrema on the fly. Returns a tuple containing the reconstructed phase vector and the quality status.
"""
function extract_absolute_phase_from_fresh_sweep(raw_profile::Vector{Float32})
    num_pts = length(raw_profile)
    measured_phase = fill(NaN32, num_pts)
    
    smoothed = smooth_profile_validation(raw_profile, 2)
    
    local_mins = Int[]
    local_maxes = Int[]
    
    for idx in 3:(num_pts - 2)
        if smoothed[idx] < smoothed[idx-1] && smoothed[idx] <= smoothed[idx+1]
            push!(local_mins, idx)
        elseif smoothed[idx] > smoothed[idx-1] && smoothed[idx] >= smoothed[idx+1]
            push!(local_maxes, idx)
        end
    end
    
    try
        # Enforce chronological peak sorting to parse the 4 intervals cleanly
        m1_idx = local_mins[findfirst(m -> m >= 15, local_mins)]
        max1_idx = local_maxes[findfirst(mx -> mx > m1_idx, local_maxes)]
        m2_idx = local_mins[findfirst(m -> m > max1_idx, local_mins)]
        max2_idx = local_maxes[findfirst(mx -> mx > m2_idx, local_maxes)]
        m3_idx = local_mins[findfirst(m -> m > max2_idx + 5, local_mins)]
        
        if m1_idx < max1_idx && max1_idx < m2_idx && m2_idx < max2_idx && max2_idx < m3_idx
            # Interpolate exact sub-pixel extrema locations
            v_m1   = fit_validation_vertex(m1_idx, smoothed)
            v_max1 = fit_validation_vertex(max1_idx, smoothed)
            v_m2   = fit_validation_vertex(m2_idx, smoothed)
            v_max2 = fit_validation_vertex(max2_idx, smoothed)
            v_m3   = fit_validation_vertex(m3_idx, smoothed)
            
            segments = [1:m1_idx, (m1_idx+1):max1_idx, (max1_idx+1):m2_idx, (m2_idx+1):max2_idx, (max2_idx+1):num_pts]
            
            # Sub-divide and unroll based on native waveform slope directions
            for s in 1:5
                idx_range = segments[s]
                length(idx_range) == 0 && continue
                
                # Assign baseline mappings derived on the fly from the active sweep
                if s == 1 # Pre-Min1 baseline
                    i_min, i_max = smoothed[m1_idx], smoothed[max1_idx]
                    for g_idx in idx_range
                        norm_i = clamp((raw_profile[g_idx] - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
                        measured_phase[g_idx] = 2.0f0 * asin(sqrt(norm_i))
                    end
                elseif s == 2 # Min1 to Max1 (Rising)
                    i_min, i_max = smoothed[m1_idx], smoothed[max1_idx]
                    for g_idx in idx_range
                        norm_i = clamp((raw_profile[g_idx] - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
                        measured_phase[g_idx] = 2.0f0 * asin(sqrt(norm_i))
                    end
                elseif s == 3 # Max1 to Min2 (Falling)
                    i_min, i_max = smoothed[m2_idx], smoothed[max1_idx]
                    for g_idx in idx_range
                        norm_i = clamp((raw_profile[g_idx] - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
                        measured_phase[g_idx] = Float32(π) + 2.0f0 * acos(sqrt(norm_i))
                    end
                elseif s == 4 # Min2 to Max2 (Rising)
                    i_min, i_max = smoothed[m2_idx], smoothed[max2_idx]
                    for g_idx in idx_range
                        norm_i = clamp((raw_profile[g_idx] - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
                        measured_phase[g_idx] = Float32(2π) + 2.0f0 * asin(sqrt(norm_i))
                    end
                elseif s == 5 # Max2 to Min3 (Falling)
                    i_min, i_max = smoothed[m3_idx], smoothed[max2_idx]
                    for g_idx in idx_range
                        norm_i = clamp((raw_profile[g_idx] - i_min) / (i_max - i_min + 1e-6), 0.0f0, 1.0f0)
                        measured_phase[g_idx] = Float32(3π) + 2.0f0 * acos(sqrt(norm_i))
                    end
                end
            end
            return measured_phase, 4 # Successfully processed 4π pixel
        else
            return fill(NaN32, num_pts), 0
        end
    catch
        return fill(NaN32, num_pts), 0 # Label corrupted or non-responsive pixels as dark spots
    end
end

# =========================================================================
# 3. MASTER INTERFEROMETER SWEEP DRIVER
# =========================================================================
function run_independent_validation_sweep(lut_generation_function, update_quality_map::Bool=false)
    intensity_cube = zeros(Float32, Nx, Ny, num_steps)
    reconstructed_phase_cube = fill(NaN32, Nx, Ny, num_steps)
    
    for p_idx in 1:num_steps
        phi_cmd = phase_axis[p_idx]
        active_frame = copy(phase_dark)
        
        for yi in 1:Ny, xi in 1:Nx
            if pixel_quality_map[xi, yi] == 4
                g_val = lut_generation_function(xi, yi, p_idx, phi_cmd)
                active_frame[slm_x_range[xi], slm_y_range[yi]] = clamp(g_val, 0.0f0, 255.0f0) / 255.0
            end
        end
        
        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)
        cam_frame = try ThorCamCSC.capture(test_cam) finally ThorCamCSC.disarmcamera(test_cam) end
        cam_img = Float32.(cam_frame)
        
        for yi in 1:Ny, xi in 1:Nx
            if pixel_quality_map[xi, yi] == 4
                intensity_cube[xi, yi, p_idx] = gaussian_weighted_sample(
                    cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi], 
                    σ_col_final, σ_row_final, kh_col, kh_row
                )
            end
        end
    end

    # Stage 1 X-Axis Fix: Now plotting against Commanded Phase
    test_xi, test_yi = 50, 50 # Pick a center pixel
    raw_profile = intensity_cube[test_xi, test_yi, :]

    fig_diag = Figure(size=(600, 400))
    ax = CM.Axis(fig_diag[1, 1], 
        title="Raw Intensity Profile (Pixel 50,50)", 
        xlabel="Commanded Phase (Radians)", 
        ylabel="Intensity (ADU)"
    )
    lines!(ax, phase_axis, raw_profile, color=:black)
    display(fig_diag)

    #rest of diagonstics
    println("Generating Unrolling Diagnostics...")

    # Pick 5 random valid pixels for Graph 1, and isolate the first one for the deep dive
    valid_coords = [(xi, yi) for xi in 1:Nx, yi in 1:Ny if pixel_quality_map[xi, yi] == 4]
    sample_pixels = shuffle(valid_coords)[1:5]
    target_xi, target_yi = sample_pixels[1]

    target_profile = intensity_cube[target_xi, target_yi, :]
    target_smoothed = smooth_profile_validation(target_profile, 2)
    measured_phase_target, _ = extract_absolute_phase_from_fresh_sweep(target_profile)

    # Find extrema for the specific target pixel to visualize segments
    l_mins = Int[]
    l_maxes = Int[]
    for idx in 3:(num_steps-2)
        if target_smoothed[idx] < target_smoothed[idx-1] && target_smoothed[idx] <= target_smoothed[idx+1]
            push!(l_mins, idx)
        elseif target_smoothed[idx] > target_smoothed[idx-1] && target_smoothed[idx] >= target_smoothed[idx+1]
            push!(l_maxes, idx)
        end
    end

    fig_minisets = Figure(size=(1200, 800), font="Arial")

    # ---------------------------------------------------------
    # Graph 1: 5 Random Points with Extrema Labeled
    # ---------------------------------------------------------
    ax1 = CM.Axis(fig_minisets[1, 1],
        title="Graph 1: 5 Random Pixels (Extrema Check)",
        xlabel="Commanded Phase", ylabel="Intensity"
    )
    colors = [:black, :gray40, :gray60, :gray70, :gray80]
    for (i, (xi, yi)) in enumerate(sample_pixels)
        prof = smooth_profile_validation(intensity_cube[xi, yi, :], 2)
        lines!(ax1, phase_axis, prof, color=colors[i], linewidth=1.5)

        # Locate and scatter extrema
        c_mins = [idx for idx in 3:(num_steps-2) if prof[idx] < prof[idx-1] && prof[idx] <= prof[idx+1]]
        c_maxes = [idx for idx in 3:(num_steps-2) if prof[idx] > prof[idx-1] && prof[idx] >= prof[idx+1]]

        scatter!(ax1, phase_axis[c_mins], prof[c_mins], color=:blue, markersize=8)
        scatter!(ax1, phase_axis[c_maxes], prof[c_maxes], color=:red, markersize=8)
    end

    # ---------------------------------------------------------
    # Graph 2: Random Point Segmented
    # ---------------------------------------------------------
    ax2 = CM.Axis(fig_minisets[1, 2],
        title="Graph 2: Segment Isolation (Pixel $target_xi, $target_yi)",
        xlabel="Commanded Phase", ylabel="Intensity"
    )
    lines!(ax2, phase_axis, target_profile, color=:lightgray, linewidth=2, label="Raw")

    # Safely attempt to parse the 5 segments if enough extrema exist
    if length(l_mins) >= 3 && length(l_maxes) >= 2
        m1, m2, m3 = l_mins[1], l_mins[2], l_mins[3]
        mx1, mx2 = l_maxes[1], l_maxes[2]

        seg1 = 1:m1
        seg2 = m1:mx1
        seg3 = mx1:m2
        seg4 = m2:mx2
        seg5 = mx2:num_steps

        lines!(ax2, phase_axis[seg1], target_profile[seg1], color=:purple, linewidth=3, label="Pre-Min")
        lines!(ax2, phase_axis[seg2], target_profile[seg2], color=:dodgerblue, linewidth=3, label="Rise 1")
        lines!(ax2, phase_axis[seg3], target_profile[seg3], color=:forestgreen, linewidth=3, label="Fall 1")
        lines!(ax2, phase_axis[seg4], target_profile[seg4], color=:darkorange, linewidth=3, label="Rise 2")
        lines!(ax2, phase_axis[seg5], target_profile[seg5], color=:crimson, linewidth=3, label="Fall 2")
        axislegend(ax2, position=:rt)
    else
        text!(ax2, 0.5, 0.5, text="Failed to find required extrema\nfor segmentation", color=:red, align=(:center, :center))
    end

    # ---------------------------------------------------------
    # Graph 3: Intensity Converted to Phase
    # ---------------------------------------------------------
    ax3 = CM.Axis(fig_minisets[2, 1],
        title="Graph 3: Unrolled Absolute Phase",
        xlabel="Step Index", ylabel="Measured Phase (rad)"
    )
    lines!(ax3, 1:num_steps, measured_phase_target, color=:midnightblue, linewidth=2.5)

    # ---------------------------------------------------------
    # Graph 4: Measured vs Commanded Phase
    # ---------------------------------------------------------
    ax4 = CM.Axis(fig_minisets[2, 2],
        title="Graph 4: Measured Phase vs Commanded Phase",
        xlabel="Commanded Phase (rad)", ylabel="Measured Phase (rad)"
    )
    # Ideal 1:1 Reference Line
    lines!(ax4, phase_axis, phase_axis, color=:gray, linestyle=:dash, linewidth=2, label="Ideal y=x")
    lines!(ax4, phase_axis, measured_phase_target, color=:darkred, linewidth=2.5, label="Measured Response")
    axislegend(ax4, position=:lt)

    display(fig_minisets)
    
    for yi in 1:Ny, xi in 1:Nx
        pixel_quality_map[xi, yi] != 4 && continue
        ph_curve, q_status = extract_absolute_phase_from_fresh_sweep(intensity_cube[xi, yi, :])
        reconstructed_phase_cube[xi, yi, :] = ph_curve
        if update_quality_map
            validation_quality_map[xi, yi] = q_status
        end
    end
    
    rms_out  = zeros(Float32, num_steps)
    dphi_out = zeros(Float32, num_steps)
    for p_idx in 1:num_steps
        spatial_slice = filter(!isnan, reconstructed_phase_cube[:, :, p_idx])
        if length(spatial_slice) > 10
            rms_out[p_idx]  = (std(spatial_slice) / (2.0f0 * π)) * 1000.0f0
            dphi_out[p_idx] = mean(spatial_slice) - phase_axis[p_idx]
        end
    end
    return rms_out, dphi_out
end

# =========================================================================
# 4. EXECUTION TIMELINE RUNNER
# =========================================================================

println("\n>>> Condition 1: Running Manufacturer Default Linear Sweep...")
mfg_lut_func(xi, yi, p_idx, phi_cmd) = (phi_cmd / (4.0f0 * π)) * 255.0f0
mfg_rms, mfg_dphi = run_independent_validation_sweep(mfg_lut_func)

println("\n>>> Condition 2: Running Linear Grayscale Sweep (Bounded Gray Values)...")
lin_v_lut_func(xi, yi, p_idx, phi_cmd) = min1_map[xi, yi] + ((p_idx - 1) / (num_steps - 1)) * (min3_map[xi, yi] - min1_map[xi, yi])
lin_v_rms, lin_v_dphi = run_independent_validation_sweep(lin_v_lut_func)

println("\n>>> Condition 3: Running Global Calibrated Sweep (Spatial Average)...")
global_lut_func(xi, yi, p_idx, phi_cmd) = global_calib_lut[p_idx]
global_rms, global_dphi = run_independent_validation_sweep(global_lut_func)

println("\n>>> Condition 4: Running Custom Regional LUT Sweep [T = 0 Minutes]...")
regional_lut_func(xi, yi, p_idx, phi_cmd) = regional_lut_matrix[xi, yi, p_idx]
regional_t0_rms, regional_t0_dphi = run_independent_validation_sweep(regional_lut_func, true)

println("\n>>> Entering 45-Minute Environmental Stability Hold...")
sleep(10 * 60.0) #10 min rn, chage to 45 later

println("\n>>> Condition 5: Running Custom Regional LUT Sweep [T = 45 Minutes]...")
regional_t45_rms, regional_t45_dphi = run_independent_validation_sweep(regional_lut_func)

# =========================================================================
# 5. CONSOLE STATISTICS REPORTING
# =========================================================================
println("\n=========================================================================")
println("      METRIC SUMMARY EVALUATION (FILTERED OUT TURNING-POINT NOISE)       ")
println("=========================================================================")
@printf("Condition 1: Manufacturer LUT        -> Mean W_rms: %6.2f mλ | Mean Δϕ: %6.3f rad\n", mean(mfg_rms[singularity_mask]), mean(mfg_dphi[singularity_mask]))
@printf("Condition 2: Linear Grayscale LUT    -> Mean W_rms: %6.2f mλ | Mean Δϕ: %6.3f rad\n", mean(lin_v_rms[singularity_mask]), mean(lin_v_dphi[singularity_mask]))
@printf("Condition 3: Global Calibrated LUT   -> Mean W_rms: %6.2f mλ | Mean Δϕ: %6.3f rad\n", mean(global_rms[singularity_mask]), mean(global_dphi[singularity_mask]))
@printf("Condition 4: Custom Regional (T=0)   -> Mean W_rms: %6.2f mλ | Mean Δϕ: %6.3f rad\n", mean(regional_t0_rms[singularity_mask]), mean(regional_t0_dphi[singularity_mask]))
@printf("Condition 5: Custom Regional (T=45)  -> Mean W_rms: %6.2f mλ | Mean Δϕ: %6.3f rad\n", mean(regional_t45_rms[singularity_mask]), mean(regional_t45_dphi[singularity_mask]))
println("=========================================================================")

# =========================================================================
# 6. PAPER-GRADE PLOT GENERATION (FIGURES 2F, 2G, & 2H)
# =========================================================================
println("\nGenerating final figures...")

# --- FIGURE 2F: Wavefront RMS ---
fig_2F = Figure(size = (750, 520), font = "Arial")
ax_2F = CM.Axis(fig_2F[1, 1],
    title = "Figure 2F: Wavefront Spatial Aberration Comparison",
    xlabel = "Intended Phase Retardation (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"])
)
lines!(ax_2F, phase_axis, mfg_rms,          color = :black,       linewidth = 1.5, linestyle = :dash, label = "Manufacturer LUT")
lines!(ax_2F, phase_axis, lin_v_rms,        color = :darkorange,  linewidth = 2.0, linestyle = :dot,  label = "Linear Grayscale LUT")
lines!(ax_2F, phase_axis, global_rms,       color = :forestgreen, linewidth = 2.0, linestyle = :dash, label = "Global Calibrated LUT")
lines!(ax_2F, phase_axis, regional_t0_rms,  color = :deepskyblue, linewidth = 2.5, label = "Custom Regional LUT (T=0)")
lines!(ax_2F, phase_axis, regional_t45_rms, color = :crimson,     linewidth = 2.0, label = "Custom Regional LUT (T=45)")
axislegend(ax_2F, position = :rt, framevisible = true, bgcolor = :white)
ylims!(ax_2F, 0, max(maximum(regional_t0_rms[singularity_mask]) * 1.5, 45.0))
display(fig_2F)

# --- FIGURE 2G: SLM Imaging Path Active Pupil Map ---
fig_2G = Figure(size = (650, 550), font = "Arial")
ax_2G = CM.Axis(fig_2G[1, 1],
    title = "Figure 2G: Active SLM Light Path Quality Layout",
    xlabel = "SLM X Coordinates",
    ylabel = "SLM Y Coordinates",
    aspect = DataAspect()
)
# Heatmap uses a binary-like visualization where 0 (bad pixels) shows up as dark spots
heatmap!(ax_2G, slm_x_range, slm_y_range, validation_quality_map, colormap = :thermal)
display(fig_2G)

# --- FIGURE 2H: Spatial Average Phase Discrepancy ---
fig_2H = Figure(size = (750, 520), font = "Arial")
ax_2H = CM.Axis(fig_2H[1, 1],
    title = "Figure 2H: Phase Discrepancy (Measured - Intended)",
    xlabel = "Intended Phase Retardation (Radians)",
    ylabel = "Phase Deviation Δϕ [rad]",
    xticks = (0:π:4π, ["0", "π", "2π", "3π", "4π"]),
    yticks = (-π:π/4:π, ["-π", "-¾π", "-½π", "-¼π", "0", "¼π", "½π", "¾π", "π"])
)
hlines!(ax_2H, [0.0], color = :gray, linestyle = :solid, linewidth = 1.0)
lines!(ax_2H, phase_axis, mfg_dphi,          color = :black,       linewidth = 1.5, linestyle = :dash, label = "Manufacturer LUT")
lines!(ax_2H, phase_axis, lin_v_dphi,        color = :darkorange,  linewidth = 2.0, linestyle = :dot,  label = "Linear Grayscale LUT")
lines!(ax_2H, phase_axis, global_dphi,       color = :forestgreen, linewidth = 2.0, linestyle = :dash, label = "Global Calibrated LUT")
lines!(ax_2H, phase_axis, regional_t0_dphi,  color = :deepskyblue, linewidth = 2.5, label = "Custom Regional LUT (T=0)")
lines!(ax_2H, phase_axis, regional_t45_dphi, color = :crimson,     linewidth = 2.0, label = "Custom Regional LUT (T=45)")
axislegend(ax_2H, position = :rb, framevisible = true, bgcolor = :white)
ylims!(ax_2H, -π, π)
display(fig_2H)

println("Verification suite executed. Figures cleanly updated.")
  

display(run_figure_E_validation(
    regional_lut_matrix, pixel_quality_map, 
    min1_map, min3_map, affine_matrix,
    slm_x_range, slm_y_range, 
    σ_col_final, σ_row_final, phase_dark
))



















#@@@@@@@@@@@@@@@@@################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#################
#7th version lol
intensity_cube_13 = zeros(Float32, num_slm_x, num_slm_y, 256)

t_sweep_start = time()

for v in 0:255
    active_frame = copy(phase_dark)

    for yi in 1:Ny, xi in 1:Nx
        xs = slm_x_range[xi]
        ys = slm_y_range[yi]
        if pixel_quality_map[xi, yi] == 4
            # Space gray values non-linearly using the localized calibration curve
            g_val = regional_lut_matrix[xi, yi, v+1]

            
            active_frame[xs, ys] = g_val / 255.0
        else
            active_frame[xs, ys] = (v+1) / 255.0
        end
    end

    slm.phase = active_frame
    Meadowlark.writesingleimage(slm)

    cam_frame = try
        ThorCamCSC.capture(test_cam)
    finally
        ThorCamCSC.disarmcamera(test_cam)
    end
    cam_img = Float32.(cam_frame)

    for xi in 1:num_slm_x
        for yi in 1:num_slm_y
            intensity_cube_13[xi, yi, v+1] = gaussian_weighted_sample(
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
            v, elapsed, elapsed / max(v, 1) * (255 - v))
    end
end
 
total_time = round(time() - t_sweep_start, digits=1)
println("\nMethod C sweep complete in $(total_time) s.")
println("intensity_cube_13 shape: ", size(intensity_cube_13))
println("Ready for Step 8 (extrema detection).")
 
# Quick sanity plot — centre pixel curve
mid_xi = num_slm_x ÷ 2
mid_yi = num_slm_y ÷ 2
curve_demo = Float64.(intensity_cube_13[mid_xi, mid_yi, :])
 
fig_demo = Figure(size=(800, 400))
ax_demo  = CM.Axis(fig_demo[1,1],
    title  = "Method C — centre pixel ($(slm_x_range[mid_xi]), $(slm_y_range[mid_yi]))",
    xlabel = "Voltage step",
    ylabel = "Gaussian-weighted intensity (ADU)")
lines!(ax_demo, 0:255, curve_demo, color=:crimson, linewidth=2)
display(fig_demo)




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
    curve = intensity_cube_13[xi, yi, :]
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



# Analytical 3-point parabolic interpolation helper (SAFE VERSION)
function safe_fit_parabolic_vertex(idx::Int, profile::AbstractVector)
    # If the extreme is exactly on the boundary, we cannot fit a parabola.
    if idx <= 1 || idx >= length(profile)
        return Float64(idx)
    end
    
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

# Make sure you are using the safe_fit_parabolic_vertex helper I provided earlier!
# Define dimensions from your Step 13 intensity_cube
Nx_13, Ny_13, N_voltages_13 = size(intensity_cube_13)

min1_map_13 = zeros(Float64, Nx_13, Ny_13)
max1_map_13 = zeros(Float64, Nx_13, Ny_13)
min2_map_13 = zeros(Float64, Nx_13, Ny_13)
max2_map_13 = zeros(Float64, Nx_13, Ny_13)
min3_map_13 = zeros(Float64, Nx_13, Ny_13)
 
# Quality control tracking map: 4 = Full 4π, 3 = 3π Fallback, 2 = 2π Fallback, 0 = Bad/Noisy
pixel_quality_map_13 = zeros(Int, Nx_13, Ny_13)

println("Beginning Step 13: Topological Extrema Extraction with Distance Hopping...")

for xi in 1:Nx_13
    for yi in 1:Ny_13
        raw_profile = intensity_cube_13[xi, yi, :]
        smoothed = smooth_profile(raw_profile, 4) 
        
        local_mins = Int[]
        local_maxes = Int[]
        
        # --- Boundary Check: Left Edge ---
        if smoothed[1] < smoothed[2] && smoothed[1] < smoothed[3]
            push!(local_mins, 1)
        elseif smoothed[1] > smoothed[2] && smoothed[1] > smoothed[3]
            push!(local_maxes, 1)
        end
        
        # --- Interior Check ---
        for idx in 2:255
            if smoothed[idx] < smoothed[idx-1] && smoothed[idx] <= smoothed[idx+1]
                push!(local_mins, idx)
            elseif smoothed[idx] > smoothed[idx-1] && smoothed[idx] >= smoothed[idx+1]
                push!(local_maxes, idx)
            end
        end
        
        # --- Boundary Check: Right Edge ---
        if smoothed[256] < smoothed[255] && smoothed[256] < smoothed[254]
            push!(local_mins, 256)
        elseif smoothed[256] > smoothed[255] && smoothed[256] > smoothed[254]
            push!(local_maxes, 256)
        end

        # --- Topology Extraction (Distance Hopping) ---
        try
            min_dist = 25 # A true phase shift takes at least 40+ steps, 25 safely jumps over local noise wiggles
            
            # Find Min1 (The absolute first minimum detected)
            m1_idx = local_mins[1] 
            
            # Find Max1
            idx_max1 = findfirst(mx -> mx > m1_idx + min_dist, local_maxes)
            isnothing(idx_max1) && error()
            max1_idx = local_maxes[idx_max1]
            
            # Find Min2
            idx_min2 = findfirst(m -> m > max1_idx + min_dist, local_mins)
            isnothing(idx_min2) && error()
            m2_idx = local_mins[idx_min2]
            
            # Find Max2 (May not exist if curve is weak)
            idx_max2 = findfirst(mx -> mx > m2_idx + min_dist, local_maxes)
            
            # Find Min3 (May not exist if curve ends on Max2)
            idx_min3 = isnothing(idx_max2) ? nothing : findfirst(m -> m > local_maxes[idx_max2] + min_dist, local_mins)
            
            # --- Store Base Locations (Guaranteed for at least 2π) ---
            min1_map_13[xi, yi] = safe_fit_parabolic_vertex(m1_idx, smoothed)
            max1_map_13[xi, yi] = safe_fit_parabolic_vertex(max1_idx, smoothed)
            min2_map_13[xi, yi] = safe_fit_parabolic_vertex(m2_idx, smoothed)

            # --- Classify and Store Tails ---
            if !isnothing(idx_max2) && !isnothing(idx_min3)
                max2_map_13[xi, yi] = safe_fit_parabolic_vertex(local_maxes[idx_max2], smoothed)
                min3_map_13[xi, yi] = safe_fit_parabolic_vertex(local_mins[idx_min3], smoothed)
                pixel_quality_map_13[xi, yi] = 4 # Golden 4π pixel
            elseif !isnothing(idx_max2)
                max2_map_13[xi, yi] = safe_fit_parabolic_vertex(local_maxes[idx_max2], smoothed)
                pixel_quality_map_13[xi, yi] = 3 # 3π Fallback (Like your Orange curve)
            else
                pixel_quality_map_13[xi, yi] = 2 # 2π Fallback
            end
            
        catch
            pixel_quality_map_13[xi, yi] = 0
        end
    end
end


# Display diagnostic summary
total_pixels = Nx_13 * Ny_13
p4_count = count(q -> q == 4, pixel_quality_map_13)
p2_count = count(q -> q == 2, pixel_quality_map_13)
failed_count = count(q -> q == 0, pixel_quality_map_13)

println("--- Extrema Detection Summary ---")
println("Total Spatial Pixels Processed: $total_pixels")
println("Successfully Mapped 4π Pixels : $p4_count ($(round(p4_count/total_pixels*100, digits=2))%)")
println("Fallback 2π Modulated Pixels  : $p2_count ($(round(p2_count/total_pixels*100, digits=2))%)")
println("Failed / Dead Pixels Flagged  : $failed_count ($(round(failed_count/total_pixels*100, digits=2))%)")



#Plot to see 3 random slm pixles with max and min now labled correctly. 
n_pixels_to_plot = 3
rng_xi_13 = rand(1:num_slm_x, n_pixels_to_plot)
rng_yi_13 = rand(1:num_slm_y, n_pixels_to_plot)

fig = Figure(size=(1000, 600))
ax = CM.Axis(
    fig[1,1],
    title = "Voltage Response with Detected Extrema",
    xlabel = "Voltage Step",
    ylabel = "Normalized Intensity",
    xticks = 0:10:255
)

colors = CM.cgrad(:turbo, n_pixels_to_plot, categorical=true)



for (i, (xi, yi)) in enumerate(zip(rng_xi_13, rng_yi_13))

    curve = intensity_cube_13[xi, yi, :]
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
        min1_map_13[xi, yi],
        max1_map_13[xi, yi],
        min2_map_13[xi, yi],
        max2_map_13[xi, yi],
        min3_map_13[xi, yi]
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



axislegend(ax, position = :rt)

display(fig)


#now need to segment this step 13 data
segmented_data_13 = [Vector{Vector{Float32}}() for xi in 1:Nx_13, yi in 1:Ny_13]

println("Beginning Step 9: Slicing and Normalizing Phase Stroke Segments via Smoothed Data...")

segment_colors = [:deepskyblue, :darkorange, :crimson, :forestgreen]
segment_labels = [
    "Segment 1 (Min1 → Max1)",
    "Segment 2 (Max1 → Min2)",
    "Segment 3 (Min2 → Max2)",
    "Segment 4 (Max2 → Min3)"
]

mid_xi = round(Int, Nx_13 / 2)
mid_yi = round(Int, Ny_13 / 2)
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

for xi in 1:Nx_13
    for yi in 1:Ny_13
        
        # 1. Generate the smoothed profile for segment extraction
        raw_profile = intensity_cube_13[xi, yi, :]
        smoothed_profile = raw_profile
        
        if pixel_quality_map_13[xi, yi] == 4
            
            boundaries = [
                min1_map_13[xi, yi],
                max1_map_13[xi, yi],
                min2_map_13[xi, yi],
                max2_map_13[xi, yi],
                min3_map_13[xi, yi]
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
                
                push!(segmented_data_13[xi, yi], norm_i)
                
                # Plot test target pixel live
                if xi == mid_xi && yi == mid_yi
                    lines!(ax_seg, norm_v, norm_i, color = segment_colors[s], linewidth = 2.5, label = segment_labels[s])
                    scatter!(ax_seg, norm_v, norm_i, color = segment_colors[s], markersize = 6)
                end
            end
            
        elseif pixel_quality_map_13[xi, yi] == 2
            # 2π Fallback handling using smoothed profiles
            boundaries = [min1_map_13[xi, yi], max1_map_13[xi, yi], min2_map_13[xi, yi]]
            
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
                
                push!(segmented_data_13[xi, yi], norm_i)
            end
        end
        
    end
end

axislegend(ax_seg, position = :rt, framevisible = true, bgcolor = (:white, 0.85))
display(fig_seg)

println("Step 9 Complete! Normalized phase curves map elegantly to a clean template.")


#now unroll step 13 phase data
unrolled_phase_data_13 = [
    Vector{NamedTuple{(:Point, :V, :Phi), Tuple{Int, Float32, Float32}}}()
    for xi in 1:Nx_13, yi in 1:Ny_13
]

# Storage for the overlay plot data (local phase accrued vs normalized voltage)
overlay_phase_curves_13 = [Vector{Vector{Float32}}() for xi in 1:Nx_13, yi in 1:Ny_13]
overlay_voltage_curves_13 = [Vector{Vector{Float32}}() for xi in 1:Nx_13, yi in 1:Ny_13]

println("Beginning Step 10: Transforming Intensity to Phase Space...")

# Loop over every pixel in the ROI
for xi in 1:Nx_13
    for yi in 1:Ny_13
        
        # Only process high-quality 4π phase stroke pixels
        if pixel_quality_map_13[xi, yi] == 4
            
            # Retrieve precise parabolic boundaries from Step 8
            boundaries = [
                min1_map_13[xi, yi],
                max1_map_13[xi, yi],
                min2_map_13[xi, yi],
                max2_map_13[xi, yi],
                min3_map_13[xi, yi]
            ]
            
            raw_profile = intensity_cube_13[xi, yi, :]
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
                push!(overlay_phase_curves_13[xi, yi], local_phi)
                push!(overlay_voltage_curves_13[xi, yi], norm_v)
                
                # Stream into global continuous data log for LUT creation
                for idx in 1:length(seg_voltages)
                    push!(unrolled_phase_data_13[xi, yi], (
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
mid_xi = round(Int, Nx_13 / 2)
mid_yi = round(Int, Ny_13 / 2)
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
    v_curve = overlay_voltage_curves_13[mid_xi, mid_yi][s]
    p_curve = overlay_phase_curves_13[mid_xi, mid_yi][s]
    
    lines!(ax_overlay, v_curve, p_curve, color = segment_colors[s], linewidth = 2.5, label = segment_labels[s])
    scatter!(ax_overlay, v_curve, p_curve, color = segment_colors[s], markersize = 5)
end
axislegend(ax_overlay, position = :rb, framevisible = true)

# Populate Panel B: Continuous absolute curve
target_unrolled = unrolled_phase_data_13[mid_xi, mid_yi]
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


#trying to graph 2.F now
# Initialize storage for the error metrics
wfe_data_13 = [Vector{Float32}() for xi in 1:Nx_13, yi in 1:Ny_13]
wfe_voltage_13 = [Vector{Float32}() for xi in 1:Nx_13, yi in 1:Ny_13]

for xi in 1:Nx_13
    for yi in 1:Ny_13
        # We only want to analyze the high-quality pixels that achieved the full 4π stroke
        if pixel_quality_map_13[xi, yi] == 4
            target_unrolled = unrolled_phase_data_13[xi, yi]
            
            for pt in target_unrolled
                v_step = pt.V
                observed_phi = pt.Phi
                
                # The mathematically ideal phase command (linear 4π stroke)
                ideal_phi = v_step * (4.0f0 * Float32(π) / 255.0f0)
                
                # Residual error in radians
                phase_error_rad = observed_phi - ideal_phi
                
                # Convert radians to waves (1λ = 2π radians)
                phase_error_waves = phase_error_rad / (2.0f0 * Float32(π))
                
                push!(wfe_data_13[xi, yi], phase_error_waves)
                push!(wfe_voltage_13[xi, yi], v_step)
            end
        end
    end
end

println("Wavefront Error calculated. Rendering Graph 2.F...")

# ==============================================================================
# Visualization: Graph 2.F
# ==============================================================================
fig_2f = Figure(size=(900, 500))
ax_2f = CM.Axis(
    fig_2f[1, 1],
    title = "Graph 2.F: Regional Wavefront Error vs. Voltage Step",
    xlabel = "Voltage Step (0 - 255)",
    ylabel = "Wavefront Error (Waves, λ)",
    xticks = 0:32:255
)

# Pick a handful of random valid pixels to overlay, plus the center pixel
n_pixels_to_plot = 5
valid_pixels = findall(==(4), pixel_quality_map_13)

# Shuffle and grab a subset to avoid cluttering the plot
plot_indices = shuffle(valid_pixels)[1:min(n_pixels_to_plot, length(valid_pixels))]

# Ensure our mid_xi / mid_yi center pixel is included if it passed QC
center_cart = CartesianIndex(mid_xi, mid_yi)
if center_cart in valid_pixels && !(center_cart in plot_indices)
    plot_indices[1] = center_cart 
end

colors_2f = CM.cgrad(:tab10, length(plot_indices), categorical=true)

for (i, idx) in enumerate(plot_indices)
    xi, yi = idx.I
    actual_x = slm_x_range[xi]
    actual_y = slm_y_range[yi]
    
    v_steps = wfe_voltage_13[xi, yi]
    wfe_waves = wfe_data_13[xi, yi]
    
    # Sort chronologically by voltage to handle any overlapping slice endpoints seamlessly
    p = sortperm(v_steps)
    v_sorted = v_steps[p]
    wfe_sorted = wfe_waves[p]
    
    lines!(
        ax_2f, 
        v_sorted, 
        wfe_sorted, 
        color = colors_2f[i], 
        linewidth = 2.5, 
        label = "Pixel ($actual_x, $actual_y)"
    )
end

# Add a dashed reference line representing a mathematically perfect zero-error calibration
hlines!(ax_2f, [0.0], color = :black, linestyle = :dash, linewidth = 2, label = "Ideal Zero Error")

axislegend(ax_2f, position = :rt, framevisible = true)

display(fig_2f)



















###8th and hopefully final version of step 13
function run_independent_validation_sweep(lut_func; is_t45 = false)
    # -------------------------------------------------------------------------
    # 1. HARDWARE SWEEP
    # -------------------------------------------------------------------------
    intensity_cube = zeros(Float32, num_slm_x, num_slm_y, 256)
    t_sweep_start = time()

    for v in 0:255
        active_frame = copy(phase_dark)

        for yi in 1:num_slm_y, xi in 1:num_slm_x
            xs = slm_x_range[xi]
            ys = slm_y_range[yi]
            
            # Apply the passed-in LUT function
            g_val = lut_func(xi, yi, v+1, v)
            
            # Bound safety
            g_val_clamped = clamp(g_val, 0.0, 255.0)
            active_frame[xs, ys] = g_val_clamped / 255.0
        end

        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)

        cam_frame = try
            ThorCamCSC.capture(test_cam)
        finally
            ThorCamCSC.disarmcamera(test_cam)
        end
        
        cam_img = Float32.(cam_frame)

        for xi in 1:num_slm_x, yi in 1:num_slm_y
            intensity_cube[xi, yi, v+1] = gaussian_weighted_sample(
                cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi],
                σ_col_final, σ_row_final, kh_col, kh_row
            )
        end
    end

    # -------------------------------------------------------------------------
    # 2. EXTREMA DETECTION & QUALITY MAPPING
    # -------------------------------------------------------------------------
    min1_map = zeros(Float64, num_slm_x, num_slm_y)
    max1_map = zeros(Float64, num_slm_x, num_slm_y)
    min2_map = zeros(Float64, num_slm_x, num_slm_y)
    max2_map = zeros(Float64, num_slm_x, num_slm_y)
    min3_map = zeros(Float64, num_slm_x, num_slm_y)
    qual_map = zeros(Int, num_slm_x, num_slm_y)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        raw_profile = intensity_cube[xi, yi, :]
        smoothed = smooth_profile(raw_profile, 4) 
        
        local_mins = Int[]
        local_maxes = Int[]
        
        
        # --- Boundary Check: Left Edge ---
        if smoothed[1] < smoothed[2] && smoothed[1] < smoothed[3]
            push!(local_mins, 1)
        elseif smoothed[1] > smoothed[2] && smoothed[1] > smoothed[3]
            push!(local_maxes, 1)
        end
        
        # --- Interior Check ---
        for idx in 2:255
            if smoothed[idx] < smoothed[idx-1] && smoothed[idx] <= smoothed[idx+1]
                push!(local_mins, idx)
            elseif smoothed[idx] > smoothed[idx-1] && smoothed[idx] >= smoothed[idx+1]
                push!(local_maxes, idx)
            end
        end
        
        # --- Boundary Check: Right Edge ---
        if smoothed[256] < smoothed[255] && smoothed[256] < smoothed[254]
            push!(local_mins, 256)
        elseif smoothed[256] > smoothed[255] && smoothed[256] > smoothed[254]
            push!(local_maxes, 256)
        end

        try
            min_dist = 25
            m1_idx = local_mins[1] 
            
            idx_max1 = findfirst(mx -> mx > m1_idx + min_dist, local_maxes)
            isnothing(idx_max1) && error()
            max1_idx = local_maxes[idx_max1]
            
            idx_min2 = findfirst(m -> m > max1_idx + min_dist, local_mins)
            isnothing(idx_min2) && error()
            m2_idx = local_mins[idx_min2]
            
            idx_max2 = findfirst(mx -> mx > m2_idx + min_dist, local_maxes)
            idx_min3 = isnothing(idx_max2) ? nothing : findfirst(m -> m > local_maxes[idx_max2] + min_dist, local_mins)
            
            min1_map[xi, yi] = safe_fit_parabolic_vertex(m1_idx, smoothed)
            max1_map[xi, yi] = safe_fit_parabolic_vertex(max1_idx, smoothed)
            min2_map[xi, yi] = safe_fit_parabolic_vertex(m2_idx, smoothed)

            if !isnothing(idx_max2) && !isnothing(idx_min3)
                max2_map[xi, yi] = safe_fit_parabolic_vertex(local_maxes[idx_max2], smoothed)
                min3_map[xi, yi] = safe_fit_parabolic_vertex(local_mins[idx_min3], smoothed)
                qual_map[xi, yi] = 4 
            elseif !isnothing(idx_max2)
                qual_map[xi, yi] = 3 
            else
                qual_map[xi, yi] = 2 
            end
        catch
            qual_map[xi, yi] = 0
        end
    end

    # -------------------------------------------------------------------------
    # 3. PHASE UNROLLING & mλ ERROR CALCULATION
    # -------------------------------------------------------------------------
    wfe_accumulator = zeros(Float32, 256)
    wfe_counts = zeros(Int, 256)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        if qual_map[xi, yi] == 4
            boundaries = [min1_map[xi, yi], max1_map[xi, yi], min2_map[xi, yi], max2_map[xi, yi], min3_map[xi, yi]]
            smoothed_profile = smooth_profile(intensity_cube[xi, yi, :], 2)
            phase_baselines = [0.0f0, Float32(π), Float32(2π), Float32(3π)]
            
            for s in 1:4
                start_idx = round(Int, boundaries[s])
                end_idx   = round(Int, boundaries[s+1])
                
                seg_intensities = smoothed_profile[start_idx:end_idx]
                seg_voltages    = Float32.( (start_idx-1):(end_idx-1) )
                
                i_min, i_max = extrema(seg_intensities)
                norm_i = (i_max > i_min) ? (seg_intensities .- i_min) ./ (i_max - i_min) : zeros(Float32, length(seg_intensities))
                
                local_phi = (s == 1 || s == 3) ? 2.0f0 .* asin.(sqrt.(norm_i)) : 2.0f0 .* acos.(sqrt.(norm_i))
                abs_phi = local_phi .+ phase_baselines[s]
                
                # Calculate WFE in milli-lambda for each point in the segment
                for (idx, v_step) in enumerate(seg_voltages)
                    ideal_phi = v_step * (4.0f0 * Float32(π) / 255.0f0)
                    phase_error_rad = abs_phi[idx] - ideal_phi
                    
                    # Convert to mλ
                    phase_error_mlambda = (phase_error_rad / (2.0f0 * Float32(π))) * 1000.0f0
                    
                    # Bin by voltage step (0-255 maps to index 1-256)
                    v_idx = round(Int, v_step) + 1
                    wfe_accumulator[v_idx] += phase_error_mlambda
                    wfe_counts[v_idx] += 1
                end
            end
        end
    end

    # Calculate the spatial average WFE across the SLM
    avg_wfe_curve = wfe_accumulator ./ max.(wfe_counts, 1)
    
    # Calculate a single RMS scalar for this run
    rms_val = sqrt(sum(avg_wfe_curve.^2) / 256.0)
    println("Independent Validation Sweep Complete. RMS WFE: $(round(rms_val, digits=4)) mλ")

    return rms_val, avg_wfe_curve
end


println("\n>>> Condition 1: Running Custom Regional LUT Sweep [T = 0 Minutes]...")
regional_lut_func(xi, yi, p_idx, v) = regional_lut_matrix[xi, yi, p_idx]
regional_t0_rms, regional_t0_wfe = run_independent_validation_sweep(regional_lut_func)


println("\n>>> Condition 2: Running Manufacturer Default Linear Sweep...")
Meadowlark.loadlut(MFR_LUT_PATH)
println("  slm7831_at633.lut loaded.")
mfg_lut_func(xi, yi, p_idx, v) = v  # 1:1 linear mapping
mfg_rms, mfg_wfe = run_independent_validation_sweep(mfg_lut_func)
Meadowlark.loadlut(LINEAR_LUT_PATH)

#println("\n>>> Condition 3: Running Linear Grayscale Sweep (Bounded Gray Values)...")
#lin_v_lut_func(xi, yi, p_idx, v) = min1_map[xi, yi] + (v / 255.0) * (min3_map[xi, yi] - min1_map[xi, yi])
#lin_v_rms, lin_v_wfe = run_independent_validation_sweep(lin_v_lut_func)

println("\n>>> Condition 4: Running Global Calibrated Sweep (Spatial Average)...")
# Compute the Global Calibrated LUT (The spatial average of your custom regional matrix)
global_calib_lut = zeros(Float32, num_steps)
for p in 1:num_steps
    valid_pixels = filter(v -> v > 0.0f0 && v <= 255.0f0, regional_lut_matrix[:, :, p])
    global_calib_lut[p] = isempty(valid_pixels) ? 0.0f0 : mean(valid_pixels)
end
global_lut_func(xi, yi, p_idx, v) = global_calib_lut[p_idx]
global_rms, global_wfe = run_independent_validation_sweep(global_lut_func)

println("\n>>> Entering 45-Minute Environmental Stability Hold...")
sleep(8 * 60.0)

println("\n>>> Condition 5: Running Custom Regional LUT Sweep [T = 45 Minutes]...")
regional_t45_rms, regional_t45_wfe = run_independent_validation_sweep(regional_lut_func, is_t45=true)



sleep(45 * 60.0)

println("\n>>> Condition 6: Running Custom Regional LUT Sweep [T = 60 Minutes]...")
regional_t60_rms, regional_t60_wfe = run_independent_validation_sweep(regional_lut_func, is_t45=true)




# ==============================================================================
# Visualization: Graph 2.F - Average Wavefront Error Comparison
# ==============================================================================
fig_2f = Figure(size=(1000, 600))
#=ax_2f = CM.Axis(
    fig_2f[1, 1],
    title = "Graph 2.F: Average Regional Wavefront Error vs. Voltage Step",
    xlabel = "Voltage Step (0 - 255)",
    ylabel = "Average Wavefront Error (mλ)",
    xticks = 0:32:255
)

v_axis = 0:255=#
ax_2f = CM.Axis(
    fig_2f[1, 1],
    title = "Graph 2.F: Average Regional Wavefront Error vs. Commanded Phase",
    xlabel = "Commanded Phase (Radians)",
    ylabel = "Average Wavefront Error (mλ)",
    xticks = (0:π:4π, ["0", "1π", "2π", "3π", "4π"])
)

# Convert 0-255 voltage steps to 0-4π phase
phase_axis = (0:255) .* (4.0f0 * Float32(π) / 255.0f0)

# Plot each condition
lines!(ax_2f, v_axis, mfg_wfe,         color=:gray,        linewidth=2.5, label="1. Manufacturer Default (RMS: $(round(mfg_rms, digits=1)) mλ)")
#lines!(ax_2f, v_axis, lin_v_wfe,       color=:dodgerblue,  linewidth=2.5, label="2. Bounded Linear (RMS: $(round(lin_v_rms, digits=1)) mλ)")
lines!(ax_2f, v_axis, global_wfe,      color=:darkorange,  linewidth=2.5, label="3. Global Average (RMS: $(round(global_rms, digits=1)) mλ)")
lines!(ax_2f, v_axis, regional_t0_wfe, color=:forestgreen, linewidth=3.0, label="4. Regional LUT T=0 (RMS: $(round(regional_t0_rms, digits=1)) mλ)")
lines!(ax_2f, v_axis, regional_t45_wfe,color=:crimson,     linewidth=3.0, label="5. Regional LUT T=45 (RMS: $(round(regional_t45_rms, digits=1)) mλ)", linestyle=:dash)
lines!(ax_2f, v_axis, regional_t60_wfe,color=:mediumvioletred, linewidth=3.0, label="6. Regional LUT T=60 (RMS: $(round(regional_t60_rms, digits=1)) mλ)", linestyle=:dash)

# Ideal zero-error reference line
hlines!(ax_2f, [0.0], color=:black, linestyle=:dot, linewidth=2, label="Ideal Zero Error")

axislegend(ax_2f, position = :rt, framevisible = true)
display(fig_2f)





#9th time, but very similar to 8th
function run_independent_validation_sweep(lut_func; is_t45 = false)
    # -------------------------------------------------------------------------
    # 1. HARDWARE SWEEP
    # -------------------------------------------------------------------------
    intensity_cube = zeros(Float32, num_slm_x, num_slm_y, 256)
    t_sweep_start = time()

    for v in 0:255
        active_frame = copy(phase_dark)

        for yi in 1:num_slm_y, xi in 1:num_slm_x
            xs = slm_x_range[xi]
            ys = slm_y_range[yi]
            
            # Apply the passed-in LUT function
            g_val = lut_func(xi, yi, v+1, v)
            
            # Bound safety
            g_val_clamped = clamp(g_val, 0.0, 255.0)
            active_frame[xs, ys] = g_val_clamped / 255.0
        end

        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)

        cam_frame = try
            ThorCamCSC.capture(test_cam)
        finally
            ThorCamCSC.disarmcamera(test_cam)
        end
        
        cam_img = Float32.(cam_frame)

        for xi in 1:num_slm_x, yi in 1:num_slm_y
            intensity_cube[xi, yi, v+1] = gaussian_weighted_sample(
                cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi],
                σ_col_final, σ_row_final, kh_col, kh_row
            )
        end
    end

    # -------------------------------------------------------------------------
    # 2. EXTREMA DETECTION & QUALITY MAPPING
    # -------------------------------------------------------------------------
    min1_map = zeros(Float64, num_slm_x, num_slm_y)
    max1_map = zeros(Float64, num_slm_x, num_slm_y)
    min2_map = zeros(Float64, num_slm_x, num_slm_y)
    max2_map = zeros(Float64, num_slm_x, num_slm_y)
    min3_map = zeros(Float64, num_slm_x, num_slm_y)
    qual_map = zeros(Int, num_slm_x, num_slm_y)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        raw_profile = intensity_cube[xi, yi, :]
        smoothed = smooth_profile(raw_profile, 4) 
        
        local_mins = Int[]
        local_maxes = Int[]
        
        # --- Boundary Check: Left Edge ---
        if smoothed[1] < smoothed[2] && smoothed[1] < smoothed[3]
            push!(local_mins, 1)
        elseif smoothed[1] > smoothed[2] && smoothed[1] > smoothed[3]
            push!(local_maxes, 1)
        end
        
        # --- Interior Check ---
        for idx in 2:255
            if smoothed[idx] < smoothed[idx-1] && smoothed[idx] <= smoothed[idx+1]
                push!(local_mins, idx)
            elseif smoothed[idx] > smoothed[idx-1] && smoothed[idx] >= smoothed[idx+1]
                push!(local_maxes, idx)
            end
        end
        
        # --- Boundary Check: Right Edge ---
        if smoothed[256] < smoothed[255] && smoothed[256] < smoothed[254]
            push!(local_mins, 256)
        elseif smoothed[256] > smoothed[255] && smoothed[256] > smoothed[254]
            push!(local_maxes, 256)
        end

        try
            min_dist = 25
            m1_idx = local_mins[1] 
            
            idx_max1 = findfirst(mx -> mx > m1_idx + min_dist, local_maxes)
            isnothing(idx_max1) && error()
            max1_idx = local_maxes[idx_max1]
            
            idx_min2 = findfirst(m -> m > max1_idx + min_dist, local_mins)
            isnothing(idx_min2) && error()
            m2_idx = local_mins[idx_min2]
            
            idx_max2 = findfirst(mx -> mx > m2_idx + min_dist, local_maxes)
            idx_min3 = isnothing(idx_max2) ? nothing : findfirst(m -> m > local_maxes[idx_max2] + min_dist, local_mins)
            
            min1_map[xi, yi] = safe_fit_parabolic_vertex(m1_idx, smoothed)
            max1_map[xi, yi] = safe_fit_parabolic_vertex(max1_idx, smoothed)
            min2_map[xi, yi] = safe_fit_parabolic_vertex(m2_idx, smoothed)

            if !isnothing(idx_max2) && !isnothing(idx_min3)
                max2_map[xi, yi] = safe_fit_parabolic_vertex(local_maxes[idx_max2], smoothed)
                min3_map[xi, yi] = safe_fit_parabolic_vertex(local_mins[idx_min3], smoothed)
                qual_map[xi, yi] = 4 
            elseif !isnothing(idx_max2)
                qual_map[xi, yi] = 3 
            else
                qual_map[xi, yi] = 2 
            end
        catch
            qual_map[xi, yi] = 0
        end
    end

    # -------------------------------------------------------------------------
    # 3. True Spatial RMS Wavefront Error Calculation
    # -------------------------------------------------------------------------
    wfe_sq_accumulator = zeros(Float32, 256)
    wfe_counts = zeros(Int, 256)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        if qual_map[xi, yi] == 4
            boundaries = [min1_map[xi, yi], max1_map[xi, yi], min2_map[xi, yi], max2_map[xi, yi], min3_map[xi, yi]]
            smoothed_profile = smooth_profile(intensity_cube[xi, yi, :], 2)
            phase_baselines = [0.0f0, Float32(π), Float32(2π), Float32(3π)]
            
            for s in 1:4
                start_idx = round(Int, boundaries[s])
                end_idx   = round(Int, boundaries[s+1])
                
                seg_intensities = smoothed_profile[start_idx:end_idx]
                seg_voltages    = Float32.( (start_idx-1):(end_idx-1) )
                
                i_min, i_max = extrema(seg_intensities)
                norm_i = (i_max > i_min) ? (seg_intensities .- i_min) ./ (i_max - i_min) : zeros(Float32, length(seg_intensities))
                
                local_phi = (s == 1 || s == 3) ? 2.0f0 .* asin.(sqrt.(norm_i)) : 2.0f0 .* acos.(sqrt.(norm_i))
                abs_phi = local_phi .+ phase_baselines[s]
                
                for (idx, v_step) in enumerate(seg_voltages)
                    ideal_phi = v_step * (4.0f0 * Float32(π) / 255.0f0)
                    phase_error_rad = abs_phi[idx] - ideal_phi
                    
                    # Convert to mλ
                    phase_error_mlambda = (phase_error_rad / (2.0f0 * Float32(π))) * 1000.0f0
                    
                    # Bin squared errors for true spatial RMS tracking
                    v_idx = round(Int, v_step) + 1
                    wfe_sq_accumulator[v_idx] += phase_error_mlambda^2
                    wfe_counts[v_idx] += 1
                end
            end
        end
    end

    # Complete the Root-Mean-Square calculation per voltage step
    rms_wfe_curve = sqrt.(wfe_sq_accumulator ./ max.(wfe_counts, 1))
    
    # Global scalar metric across the entire dataset
    global_rms_val = sqrt(sum(rms_wfe_curve.^2) / 256.0)
    println("Independent Validation Sweep Complete. Scalar Global RMS WFE: $(round(global_rms_val, digits=4)) mλ")

    return global_rms_val, rms_wfe_curve
end

# --- Run sweeps ---
println("\n>>> Condition 1: Running Custom Regional LUT Sweep [T = 0 Minutes]...")
regional_lut_func(xi, yi, p_idx, v) = regional_lut_matrix[xi, yi, p_idx]
regional_t0_rms, regional_t0_wfe = run_independent_validation_sweep(regional_lut_func)

println("\n>>> Condition 2: Running Manufacturer Default Linear Sweep...")
Meadowlark.loadlut(MFR_LUT_PATH)
mfg_lut_func(xi, yi, p_idx, v) = v
mfg_rms, mfg_wfe = run_independent_validation_sweep(mfg_lut_func)
Meadowlark.loadlut(LINEAR_LUT_PATH)

println("\n>>> Condition 4: Running Global Calibrated Sweep (Spatial Average)...")
global_calib_lut = zeros(Float32, num_steps)
for p in 1:num_steps
    valid_pixels = filter(v -> v > 0.0f0 && v <= 255.0f0, regional_lut_matrix[:, :, p])
    global_calib_lut[p] = isempty(valid_pixels) ? 0.0f0 : mean(valid_pixels)
end
global_lut_func(xi, yi, p_idx, v) = global_calib_lut[p_idx]
global_rms, global_wfe = run_independent_validation_sweep(global_lut_func)

#println("\n>>> Entering 45-Minute Environmental Stability Hold...")
#sleep(45 * 60.0)

#println("\n>>> Condition 5: Running Custom Regional LUT Sweep [T = 45 Minutes]...")
#regional_t45_rms, regional_t45_wfe = run_independent_validation_sweep(regional_lut_func, is_t45=true)

#sleep(15 * 60.0)

#println("\n>>> Condition 6: Running Custom Regional LUT Sweep [T = 60 Minutes]...")
#regional_t60_rms, regional_t60_wfe = run_independent_validation_sweep(regional_lut_func, is_t45=true)


# ==============================================================================
# Visualization: Graph 2.F - True RMS Wavefront Error Comparison
# ==============================================================================
fig_2f = Figure(size=(1000, 600))

ax_2f = CM.Axis(
    fig_2f[1, 1],
    title = "Graph 2.F: Spatial RMS Wavefront Error vs. Commanded Phase",
    xlabel = "Commanded Phase (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    yscale = log10, # Logarithmic layout matching literature
    xticks = (0:π:4π, ["0", "1π", "2π", "3π", "4π"])
)

# Convert 0-255 voltage steps to continuous 0-4π phase axis
phase_axis = (0:255) .* (4.0f0 * Float32(π) / 255.0f0)

# Truncation logic: Restrict the manufacturer default to the 0 to 2π domain
mfr_cutoff_idx = findlast(p -> p <= 2.0f0 * Float32(π), phase_axis)
mfr_phase_axis = phase_axis[1:mfr_cutoff_idx]
mfr_wfe_truncated = mfg_wfe[1:mfr_cutoff_idx]

# Plot profiles with corrected x-coordinates (phase_axis)
lines!(ax_2f, mfr_phase_axis, mfr_wfe_truncated, color=:gray, linewidth=2.5, label="1. Manufacturer Default (RMS: $(round(mfg_rms, digits=1)) mλ)")
lines!(ax_2f, phase_axis, global_wfe,       color=:darkorange,  linewidth=2.5, label="3. Global Average (RMS: $(round(global_rms, digits=1)) mλ)")
lines!(ax_2f, phase_axis, regional_t0_wfe,  color=:forestgreen, linewidth=3.0, label="4. Regional LUT T=0 (RMS: $(round(regional_t0_rms, digits=1)) mλ)")
#lines!(ax_2f, phase_axis, regional_t45_wfe, color=:crimson,     linewidth=3.0, label="5. Regional LUT T=45 (RMS: $(round(regional_t45_rms, digits=1)) mλ)", linestyle=:dash)
#lines!(ax_2f, phase_axis, regional_t60_wfe, color=:mediumvioletred, linewidth=3.0, label="6. Regional LUT T=60 (RMS: $(round(regional_t60_rms, digits=1)) mλ)", linestyle=:dash)

axislegend(ax_2f, position = :rt, framevisible = true)
display(fig_2f)









#10
function run_independent_validation_sweep(lut_func)
    # -------------------------------------------------------------------------
    # 1. HARDWARE SWEEP
    # -------------------------------------------------------------------------
    intensity_cube = zeros(Float32, num_slm_x, num_slm_y, 256)
    t_sweep_start = time()

    for v in 0:255
        active_frame = copy(phase_dark)

        for yi in 1:num_slm_y, xi in 1:num_slm_x
            xs = slm_x_range[xi]
            ys = slm_y_range[yi]
            
            # Apply the passed-in LUT function
            g_val = lut_func(xi, yi, v+1, v)
            
            # Bound safety
            g_val_clamped = clamp(g_val, 0.0, 255.0)
            active_frame[xs, ys] = g_val_clamped / 255.0
        end

        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)

        cam_frame = try
            ThorCamCSC.capture(test_cam)
        finally
            ThorCamCSC.disarmcamera(test_cam)
        end
        
        cam_img = Float32.(cam_frame)

        for xi in 1:num_slm_x, yi in 1:num_slm_y
            intensity_cube[xi, yi, v+1] = gaussian_weighted_sample(
                cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi],
                σ_col_final, σ_row_final, kh_col, kh_row
            )
        end
    end

    # -------------------------------------------------------------------------
    # 2. EXTREMA DETECTION & QUALITY MAPPING
    # -------------------------------------------------------------------------
    min1_map = zeros(Float64, num_slm_x, num_slm_y)
    max1_map = zeros(Float64, num_slm_x, num_slm_y)
    min2_map = zeros(Float64, num_slm_x, num_slm_y)
    max2_map = zeros(Float64, num_slm_x, num_slm_y)
    min3_map = zeros(Float64, num_slm_x, num_slm_y)
    qual_map = zeros(Int, num_slm_x, num_slm_y)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        raw_profile = intensity_cube[xi, yi, :]
        smoothed = smooth_profile(raw_profile, 4) 
        
        local_mins = Int[]
        local_maxes = Int[]
        
        # --- Boundary Check: Left Edge ---
        if smoothed[1] < smoothed[2] && smoothed[1] < smoothed[3]
            push!(local_mins, 1)
        elseif smoothed[1] > smoothed[2] && smoothed[1] > smoothed[3]
            push!(local_maxes, 1)
        end
        
        # --- Interior Check ---
        for idx in 2:255
            if smoothed[idx] < smoothed[idx-1] && smoothed[idx] <= smoothed[idx+1]
                push!(local_mins, idx)
            elseif smoothed[idx] > smoothed[idx-1] && smoothed[idx] >= smoothed[idx+1]
                push!(local_maxes, idx)
            end
        end
        
        # --- Boundary Check: Right Edge ---
        if smoothed[256] < smoothed[255] && smoothed[256] < smoothed[254]
            push!(local_mins, 256)
        elseif smoothed[256] > smoothed[255] && smoothed[256] > smoothed[254]
            push!(local_maxes, 256)
        end

        try
            min_dist = 25
            m1_idx = local_mins[1] 
            
            idx_max1 = findfirst(mx -> mx > m1_idx + min_dist, local_maxes)
            isnothing(idx_max1) && error()
            max1_idx = local_maxes[idx_max1]
            
            idx_min2 = findfirst(m -> m > max1_idx + min_dist, local_mins)
            isnothing(idx_min2) && error()
            m2_idx = local_mins[idx_min2]
            
            idx_max2 = findfirst(mx -> mx > m2_idx + min_dist, local_maxes)
            idx_min3 = isnothing(idx_max2) ? nothing : findfirst(m -> m > local_maxes[idx_max2] + min_dist, local_mins)
            
            min1_map[xi, yi] = safe_fit_parabolic_vertex(m1_idx, smoothed)
            max1_map[xi, yi] = safe_fit_parabolic_vertex(max1_idx, smoothed)
            min2_map[xi, yi] = safe_fit_parabolic_vertex(m2_idx, smoothed)

            if !isnothing(idx_max2) && !isnothing(idx_min3)
                max2_map[xi, yi] = safe_fit_parabolic_vertex(local_maxes[idx_max2], smoothed)
                min3_map[xi, yi] = safe_fit_parabolic_vertex(local_mins[idx_min3], smoothed)
                qual_map[xi, yi] = 4 
            elseif !isnothing(idx_max2)
                qual_map[xi, yi] = 3 
            else
                qual_map[xi, yi] = 2 
            end
        catch
            qual_map[xi, yi] = 0
        end
    end

    # -------------------------------------------------------------------------
    # 3. True Spatial RMS Wavefront Error Calculation
    # -------------------------------------------------------------------------
    wfe_sq_accumulator = zeros(Float32, 256)
    wfe_counts = zeros(Int, 256)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        if qual_map[xi, yi] == 4
            boundaries = [min1_map[xi, yi], max1_map[xi, yi], min2_map[xi, yi], max2_map[xi, yi], min3_map[xi, yi]]
            smoothed_profile = smooth_profile(intensity_cube[xi, yi, :], 2)
            phase_baselines = [0.0f0, Float32(π), Float32(2π), Float32(3π)]
            
            for s in 1:4
                start_idx = round(Int, boundaries[s])
                end_idx   = round(Int, boundaries[s+1])
                
                seg_intensities = smoothed_profile[start_idx:end_idx]
                seg_voltages    = Float32.( (start_idx-1):(end_idx-1) )
                
                i_min, i_max = extrema(seg_intensities)
                norm_i = (i_max > i_min) ? (seg_intensities .- i_min) ./ (i_max - i_min) : zeros(Float32, length(seg_intensities))
                
                local_phi = (s == 1 || s == 3) ? 2.0f0 .* asin.(sqrt.(norm_i)) : 2.0f0 .* acos.(sqrt.(norm_i))
                abs_phi = local_phi .+ phase_baselines[s]
                
                for (idx, v_step) in enumerate(seg_voltages)
                    ideal_phi = v_step * (4.0f0 * Float32(π) / 255.0f0)
                    phase_error_rad = abs_phi[idx] - ideal_phi
                    
                    # Convert to mλ
                    phase_error_mlambda = (phase_error_rad / (2.0f0 * Float32(π))) * 1000.0f0
                    
                    # Bin squared errors for true spatial RMS tracking
                    v_idx = round(Int, v_step) + 1
                    wfe_sq_accumulator[v_idx] += phase_error_mlambda^2
                    wfe_counts[v_idx] += 1
                end
            end
        end
    end

    # Complete the Root-Mean-Square calculation per voltage step
    rms_wfe_curve = sqrt.(wfe_sq_accumulator ./ max.(wfe_counts, 1))
    
    # Global scalar metric across the entire dataset
    global_rms_val = sqrt(sum(rms_wfe_curve.^2) / 256.0)
    println("Independent Validation Sweep Complete. Scalar Global RMS WFE: $(round(global_rms_val, digits=4)) mλ")

    return global_rms_val, rms_wfe_curve
end

# --- Run sweeps ---
println("\n>>> Condition 1: Running Custom Regional LUT Sweep [T = 0 Minutes]...")
regional_lut_func(xi, yi, p_idx, v) = regional_lut_matrix[xi, yi, p_idx]
regional_t0_rms, regional_t0_wfe = run_independent_validation_sweep(regional_lut_func)

println("\n>>> Condition 2: Running Manufacturer Default Linear Sweep...")
Meadowlark.loadlut(MFR_LUT_PATH)
mfg_lut_func(xi, yi, p_idx, v) = v
mfg_rms, mfg_wfe = run_independent_validation_sweep(mfg_lut_func)
Meadowlark.loadlut(LINEAR_LUT_PATH)

println("\n>>> Condition 4: Running Global Calibrated Sweep (Spatial Average)...")
global_calib_lut = zeros(Float32, num_steps)
for p in 1:num_steps
    valid_pixels = filter(v -> v > 0.0f0 && v <= 255.0f0, regional_lut_matrix[:, :, p])
    global_calib_lut[p] = isempty(valid_pixels) ? 0.0f0 : mean(valid_pixels)
end
global_lut_func(xi, yi, p_idx, v) = global_calib_lut[p_idx]
global_rms, global_wfe = run_independent_validation_sweep(global_lut_func)

# Time hold blocks removed for testing speed
#println("\n>>> Entering 45-Minute Environmental Stability Hold...")
#sleep(45 * 60.0)

#println("\n>>> Condition 5: Running Custom Regional LUT Sweep [T = 45 Minutes]...")
#regional_t45_rms, regional_t45_wfe = run_independent_validation_sweep(regional_lut_func, is_t45=true)

#sleep(15 * 60.0)

#println("\n>>> Condition 6: Running Custom Regional LUT Sweep [T = 60 Minutes]...")
#regional_t60_rms, regional_t60_wfe = run_independent_validation_sweep(regional_lut_func, is_t45=true)


# ==============================================================================
# Visualization: Graph 2.F - True RMS Wavefront Error Comparison
# ==============================================================================
fig_2f = Figure(size=(1000, 600))

ax_2f = CM.Axis(
    fig_2f[1, 1],
    title = "Graph 2.F: Spatial RMS Wavefront Error vs. Commanded Phase",
    xlabel = "Commanded Phase (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    yscale = log10, 
    xticks = (0:π:4π, ["0", "1π", "2π", "3π", "4π"])
)

# Phase axis mapping
phase_axis = (0:255) .* (4.0f0 * Float32(π) / 255.0f0)

# MASKING LOGIC: Cut out the extreme artificial edges (e.g., exclude ends by 0.2 rads)
valid_mask_4pi = (phase_axis .> 0.2) .& (phase_axis .< (4.0f0 * π - 0.2))

# Truncation logic for Manufacturer Default (restrict to 0 to 2π, and trim edges)
mfr_cutoff_idx = findlast(p -> p <= 2.0f0 * Float32(π), phase_axis)
mfr_phase_axis = phase_axis[1:mfr_cutoff_idx]
mfr_wfe_truncated = mfg_wfe[1:mfr_cutoff_idx]
valid_mask_2pi = (mfr_phase_axis .> 0.2) .& (mfr_phase_axis .< (2.0f0 * π - 0.2))

# Plot profiles using the masked indices to drop the zero-dives
lines!(ax_2f, mfr_phase_axis[valid_mask_2pi], mfr_wfe_truncated[valid_mask_2pi], color=:gray, linewidth=2.5, label="1. Manufacturer Default (RMS: $(round(mfg_rms, digits=1)) mλ)")
lines!(ax_2f, phase_axis[valid_mask_4pi], global_wfe[valid_mask_4pi],       color=:darkorange,  linewidth=2.5, label="3. Global Average (RMS: $(round(global_rms, digits=1)) mλ)")
lines!(ax_2f, phase_axis[valid_mask_4pi], regional_t0_wfe[valid_mask_4pi],  color=:forestgreen, linewidth=3.0, label="4. Regional LUT T=0 (RMS: $(round(regional_t0_rms, digits=1)) mλ) ")
#lines!(ax_2f, phase_axis, regional_t45_wfe, color=:crimson,     linewidth=3.0, label="5. Regional LUT T=45 (RMS: $(round(regional_t45_rms, digits=1)) mλ)", linestyle=:dash)
#lines!(ax_2f, phase_axis, regional_t60_wfe, color=:mediumvioletred, linewidth=3.0, label="6. Regional LUT T=60 (RMS: $(round(regional_t60_rms, digits=1)) mλ)", linestyle=:dash)
axislegend(ax_2f, position = :rt, framevisible = true)
display(fig_2f)






#11
function run_independent_validation_sweep(lut_func)
    # -------------------------------------------------------------------------
    # 1. HARDWARE SWEEP
    # -------------------------------------------------------------------------
    intensity_cube = zeros(Float32, num_slm_x, num_slm_y, 256)
    t_sweep_start = time()

    for v in 0:255
        active_frame = copy(phase_dark)

        for yi in 1:num_slm_y, xi in 1:num_slm_x
            xs = slm_x_range[xi]
            ys = slm_y_range[yi]
            
            g_val = lut_func(xi, yi, v+1, v)
            g_val_clamped = clamp(g_val, 0.0, 255.0)
            active_frame[xs, ys] = g_val_clamped / 255.0
        end

        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)

        cam_frame = try
            ThorCamCSC.capture(test_cam)
        finally
            ThorCamCSC.disarmcamera(test_cam)
        end
        
        cam_img = Float32.(cam_frame)

        for xi in 1:num_slm_x, yi in 1:num_slm_y
            intensity_cube[xi, yi, v+1] = gaussian_weighted_sample(
                cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi],
                σ_col_final, σ_row_final, kh_col, kh_row
            )
        end
    end

    # -------------------------------------------------------------------------
    # 2. EXTREMA DETECTION & QUALITY MAPPING
    # -------------------------------------------------------------------------
    min1_map = zeros(Float64, num_slm_x, num_slm_y)
    max1_map = zeros(Float64, num_slm_x, num_slm_y)
    min2_map = zeros(Float64, num_slm_x, num_slm_y)
    max2_map = zeros(Float64, num_slm_x, num_slm_y)
    min3_map = zeros(Float64, num_slm_x, num_slm_y)
    qual_map = zeros(Int, num_slm_x, num_slm_y)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        raw_profile = intensity_cube[xi, yi, :]
        smoothed = smooth_profile(raw_profile, 4) 
        
        local_mins = Int[]
        local_maxes = Int[]
        
        # --- Boundary Check: Left Edge ---
        if smoothed[1] < smoothed[2] && smoothed[1] < smoothed[3]
            push!(local_mins, 1)
        elseif smoothed[1] > smoothed[2] && smoothed[1] > smoothed[3]
            push!(local_maxes, 1)
        end
        
        # --- Interior Check ---
        for idx in 2:255
            if smoothed[idx] < smoothed[idx-1] && smoothed[idx] <= smoothed[idx+1]
                push!(local_mins, idx)
            elseif smoothed[idx] > smoothed[idx-1] && smoothed[idx] >= smoothed[idx+1]
                push!(local_maxes, idx)
            end
        end
        
        # --- Boundary Check: Right Edge ---
        if smoothed[256] < smoothed[255] && smoothed[256] < smoothed[254]
            push!(local_mins, 256)
        elseif smoothed[256] > smoothed[255] && smoothed[256] > smoothed[254]
            push!(local_maxes, 256)
        end

        try
            min_dist = 25
            m1_idx = local_mins[1] 
            
            idx_max1 = findfirst(mx -> mx > m1_idx + min_dist, local_maxes)
            isnothing(idx_max1) && error()
            max1_idx = local_maxes[idx_max1]
            
            idx_min2 = findfirst(m -> m > max1_idx + min_dist, local_mins)
            isnothing(idx_min2) && error()
            m2_idx = local_mins[idx_min2]
            
            idx_max2 = findfirst(mx -> mx > m2_idx + min_dist, local_maxes)
            idx_min3 = isnothing(idx_max2) ? nothing : findfirst(m -> m > local_maxes[idx_max2] + min_dist, local_mins)
            
            min1_map[xi, yi] = safe_fit_parabolic_vertex(m1_idx, smoothed)
            max1_map[xi, yi] = safe_fit_parabolic_vertex(max1_idx, smoothed)
            min2_map[xi, yi] = safe_fit_parabolic_vertex(m2_idx, smoothed)

            if !isnothing(idx_max2) && !isnothing(idx_min3)
                max2_map[xi, yi] = safe_fit_parabolic_vertex(local_maxes[idx_max2], smoothed)
                min3_map[xi, yi] = safe_fit_parabolic_vertex(local_mins[idx_min3], smoothed)
                qual_map[xi, yi] = 4 
            elseif !isnothing(idx_max2)
                qual_map[xi, yi] = 3 
            else
                qual_map[xi, yi] = 2 
            end
        catch
            qual_map[xi, yi] = 0
        end
    end

    # -------------------------------------------------------------------------
    # 3. True Spatial RMS and Phase Error Calculation
    # -------------------------------------------------------------------------
    wfe_sq_accumulator = zeros(Float32, 256)
    phase_diff_accumulator = zeros(Float32, 256) # NEW: Tracks the raw phase difference
    wfe_counts = zeros(Int, 256)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        if qual_map[xi, yi] == 4
            boundaries = [min1_map[xi, yi], max1_map[xi, yi], min2_map[xi, yi], max2_map[xi, yi], min3_map[xi, yi]]
            smoothed_profile = smooth_profile(intensity_cube[xi, yi, :], 2)
            phase_baselines = [0.0f0, Float32(π), Float32(2π), Float32(3π)]
            
            for s in 1:4
                start_idx = round(Int, boundaries[s])
                end_idx   = round(Int, boundaries[s+1])
                
                seg_intensities = smoothed_profile[start_idx:end_idx]
                seg_voltages    = Float32.( (start_idx-1):(end_idx-1) )
                
                i_min, i_max = extrema(seg_intensities)
                norm_i = (i_max > i_min) ? (seg_intensities .- i_min) ./ (i_max - i_min) : zeros(Float32, length(seg_intensities))
                
                local_phi = (s == 1 || s == 3) ? 2.0f0 .* asin.(sqrt.(norm_i)) : 2.0f0 .* acos.(sqrt.(norm_i))
                abs_phi = local_phi .+ phase_baselines[s]
                
                for (idx, v_step) in enumerate(seg_voltages)
                    ideal_phi = v_step * (4.0f0 * Float32(π) / 255.0f0)
                    phase_error_rad = abs_phi[idx] - ideal_phi
                    
                    phase_error_mlambda = (phase_error_rad / (2.0f0 * Float32(π))) * 1000.0f0
                    
                    v_idx = round(Int, v_step) + 1
                    wfe_sq_accumulator[v_idx] += phase_error_mlambda^2
                    phase_diff_accumulator[v_idx] += phase_error_rad # Accumulate the raw difference in radians
                    wfe_counts[v_idx] += 1
                end
            end
        end
    end

    rms_wfe_curve = sqrt.(wfe_sq_accumulator ./ max.(wfe_counts, 1))
    avg_phase_diff_curve = phase_diff_accumulator ./ max.(wfe_counts, 1) # Calculate the true average difference
    global_rms_val = sqrt(sum(rms_wfe_curve.^2) / 256.0)
    
    println("Independent Validation Sweep Complete. Scalar Global RMS WFE: $(round(global_rms_val, digits=4)) mλ")

    # Now returning 3 items instead of 2
    return global_rms_val, rms_wfe_curve, avg_phase_diff_curve 
end

# --- Run sweeps (Now catching 3 returned variables) ---
println("\n>>> Condition 1: Running Custom Regional LUT Sweep [T = 0 Minutes]...")
regional_lut_func(xi, yi, p_idx, v) = regional_lut_matrix[xi, yi, p_idx]
regional_t0_rms, regional_t0_wfe, regional_t0_diff = run_independent_validation_sweep(regional_lut_func)

println("\n>>> Condition 2: Running Manufacturer Default Linear Sweep...")
Meadowlark.loadlut(MFR_LUT_PATH)
mfg_lut_func(xi, yi, p_idx, v) = v
mfg_rms, mfg_wfe, mfg_diff = run_independent_validation_sweep(mfg_lut_func)
Meadowlark.loadlut(LINEAR_LUT_PATH)

println("\n>>> Condition 4: Running Global Calibrated Sweep (Spatial Average)...")
global_calib_lut = zeros(Float32, num_steps)
for p in 1:num_steps
    valid_pixels = filter(v -> v > 0.0f0 && v <= 255.0f0, regional_lut_matrix[:, :, p])
    global_calib_lut[p] = isempty(valid_pixels) ? 0.0f0 : mean(valid_pixels)
end
global_lut_func(xi, yi, p_idx, v) = global_calib_lut[p_idx]
global_rms, global_wfe, global_diff = run_independent_validation_sweep(global_lut_func)

# -----------------------------------------------------------------------------
# GRAPHING LOGIC
# -----------------------------------------------------------------------------
# Reusing your mask arrays from the 2.F graph to ensure we drop edge artifacts
phase_axis = (0:255) .* (4.0f0 * Float32(π) / 255.0f0)
valid_mask_4pi = (phase_axis .> 0.2) .& (phase_axis .< (4.0f0 * π - 0.2))

mfr_cutoff_idx = findlast(p -> p <= 2.0f0 * Float32(π), phase_axis)
mfr_phase_axis = phase_axis[1:mfr_cutoff_idx]
valid_mask_2pi = (mfr_phase_axis .> 0.2) .& (mfr_phase_axis .< (2.0f0 * π - 0.2))

fig_2f = Figure(size=(1000, 600))

ax_2f = CM.Axis(
    fig_2f[1, 1],
    title = "Graph 2.F: Spatial RMS Wavefront Error vs. Commanded Phase",
    xlabel = "Commanded Phase (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    yscale = log10, 
    xticks = (0:π:4π, ["0", "1π", "2π", "3π", "4π"])
)



# Truncation logic for Manufacturer Default (restrict to 0 to 2π, and trim edges)
mfr_wfe_truncated = mfg_wfe[1:mfr_cutoff_idx]

# Plot profiles using the masked indices to drop the zero-dives
lines!(ax_2f, mfr_phase_axis[valid_mask_2pi], mfr_wfe_truncated[valid_mask_2pi], color=:gray, linewidth=2.5, label="1. Manufacturer Default (RMS: $(round(mfg_rms, digits=1)) mλ)")
lines!(ax_2f, phase_axis[valid_mask_4pi], global_wfe[valid_mask_4pi],       color=:darkorange,  linewidth=2.5, label="3. Global Average (RMS: $(round(global_rms, digits=1)) mλ)")
lines!(ax_2f, phase_axis[valid_mask_4pi], regional_t0_wfe[valid_mask_4pi],  color=:forestgreen, linewidth=3.0, label="4. Regional LUT T=0 (RMS: $(round(regional_t0_rms, digits=1)) mλ) ")
#lines!(ax_2f, phase_axis, regional_t45_wfe, color=:crimson,     linewidth=3.0, label="5. Regional LUT T=45 (RMS: $(round(regional_t45_rms, digits=1)) mλ)", linestyle=:dash)
#lines!(ax_2f, phase_axis, regional_t60_wfe, color=:mediumvioletred, linewidth=3.0, label="6. Regional LUT T=60 (RMS: $(round(regional_t60_rms, digits=1)) mλ)", linestyle=:dash)
axislegend(ax_2f, position = :rt, framevisible = true)
display(fig_2f)



# ==============================================================================
# Visualization: Graph 2.H - Phase Discrepancy
# ==============================================================================
fig_2h = Figure(size=(1000, 600))

ax_2h = CM.Axis(
    fig_2h[1, 1],
    title = "Graph 2.H: Phase Discrepancy (Measured - Intended)",
    xlabel = "Commanded Phase (Radians)",
    ylabel = "Delta Phase Δϕ [rad]",
    xticks = (0:π:4π, ["0", "1π", "2π", "3π", "4π"]),
    yticks = (-π/4:π/8:π/4, ["-π/4", "-π/8", "0", "π/8", "π/4"]) # Spaced out from -pi/4 to pi/4
)

ylims!(ax_2h, -π/4, π/4)
xlims!(ax_2h, 0, 4π)

# Add a dashed reference line exactly at 0 error
hlines!(ax_2h, [0.0], color=:black, linestyle=:dash, linewidth=1.5)

# Plot Manufacturer Default (Truncated to 2π)
lines!(ax_2h, mfr_phase_axis[valid_mask_2pi], mfg_diff[1:mfr_cutoff_idx][valid_mask_2pi], 
       color=:gray, linewidth=2.5, label="1. Manufacturer Default")

# Plot Global Average
lines!(ax_2h, phase_axis[valid_mask_4pi], global_diff[valid_mask_4pi], 
       color=:darkorange, linewidth=2.5, label="3. Global Average")

# Plot Regional T=0
lines!(ax_2h, phase_axis[valid_mask_4pi], regional_t0_diff[valid_mask_4pi], 
       color=:forestgreen, linewidth=3.0, label="4. Regional LUT T=0")

# Commented out T=45 and T=60
# lines!(ax_2h, phase_axis[valid_mask_4pi], regional_t45_diff[valid_mask_4pi], color=:crimson, linewidth=3.0, label="5. Regional LUT T=45", linestyle=:dash)
# lines!(ax_2h, phase_axis[valid_mask_4pi], regional_t60_diff[valid_mask_4pi], color=:mediumvioletred, linewidth=3.0, label="6. Regional LUT T=60", linestyle=:dash)

axislegend(ax_2h, position = :rt, framevisible = true)
display(fig_2h)








#12
# Analytical 3-point parabolic interpolation helper (SAFE VERSION)
function safe_fit_parabolic_vertex(idx::Int, profile::AbstractVector)
    # If the extreme is exactly on the boundary, we cannot fit a parabola.
    if idx <= 1 || idx >= length(profile)
        return Float64(idx)
    end
    
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

function run_independent_validation_sweep(lut_func)
    # -------------------------------------------------------------------------
    # 1. HARDWARE SWEEP
    # -------------------------------------------------------------------------
    intensity_cube = zeros(Float32, num_slm_x, num_slm_y, 256)
    t_sweep_start = time()

    for v in 0:255
        active_frame = copy(phase_dark)

        for yi in 1:num_slm_y, xi in 1:num_slm_x
            xs = slm_x_range[xi]
            ys = slm_y_range[yi]
            
            g_val = lut_func(xi, yi, v+1, v)
            g_val_clamped = clamp(g_val, 0.0, 255.0)
            active_frame[xs, ys] = g_val_clamped / 255.0
        end

        slm.phase = active_frame
        Meadowlark.writesingleimage(slm)

        cam_frame = try
            ThorCamCSC.capture(test_cam)
        finally
            ThorCamCSC.disarmcamera(test_cam)
        end
        
        cam_img = Float32.(cam_frame)

        for xi in 1:num_slm_x, yi in 1:num_slm_y
            intensity_cube[xi, yi, v+1] = gaussian_weighted_sample(
                cam_img, xc_map_C[xi, yi], yc_map_C[xi, yi],
                σ_col_final, σ_row_final, kh_col, kh_row
            )
        end
    end

    # -------------------------------------------------------------------------
    # 2. EXTREMA DETECTION & QUALITY MAPPING
    # -------------------------------------------------------------------------
    min1_map = zeros(Float64, num_slm_x, num_slm_y)
    max1_map = zeros(Float64, num_slm_x, num_slm_y)
    min2_map = zeros(Float64, num_slm_x, num_slm_y)
    max2_map = zeros(Float64, num_slm_x, num_slm_y)
    min3_map = zeros(Float64, num_slm_x, num_slm_y)
    qual_map = zeros(Int, num_slm_x, num_slm_y)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        raw_profile = intensity_cube[xi, yi, :]
        smoothed = smooth_profile(raw_profile, 4) 
        
        local_mins = Int[]
        local_maxes = Int[]
        
        # --- Boundary Check: Left Edge ---
        if smoothed[1] < smoothed[2] && smoothed[1] < smoothed[3]
            push!(local_mins, 1)
        elseif smoothed[1] > smoothed[2] && smoothed[1] > smoothed[3]
            push!(local_maxes, 1)
        end
        
        # --- Interior Check ---
        for idx in 2:255
            if smoothed[idx] < smoothed[idx-1] && smoothed[idx] <= smoothed[idx+1]
                push!(local_mins, idx)
            elseif smoothed[idx] > smoothed[idx-1] && smoothed[idx] >= smoothed[idx+1]
                push!(local_maxes, idx)
            end
        end
        
        # --- Boundary Check: Right Edge ---
        if smoothed[256] < smoothed[255] && smoothed[256] < smoothed[254]
            push!(local_mins, 256)
        elseif smoothed[256] > smoothed[255] && smoothed[256] > smoothed[254]
            push!(local_maxes, 256)
        end

        try
            min_dist = 25
            m1_idx = local_mins[1] 
            
            idx_max1 = findfirst(mx -> mx > m1_idx + min_dist, local_maxes)
            isnothing(idx_max1) && error()
            max1_idx = local_maxes[idx_max1]
            
            idx_min2 = findfirst(m -> m > max1_idx + min_dist, local_mins)
            isnothing(idx_min2) && error()
            m2_idx = local_mins[idx_min2]
            
            idx_max2 = findfirst(mx -> mx > m2_idx + min_dist, local_maxes)
            idx_min3 = isnothing(idx_max2) ? nothing : findfirst(m -> m > local_maxes[idx_max2] + min_dist, local_mins)
            
            min1_map[xi, yi] = safe_fit_parabolic_vertex(m1_idx, smoothed)
            max1_map[xi, yi] = safe_fit_parabolic_vertex(max1_idx, smoothed)
            min2_map[xi, yi] = safe_fit_parabolic_vertex(m2_idx, smoothed)

            if !isnothing(idx_max2) && !isnothing(idx_min3)
                max2_map[xi, yi] = safe_fit_parabolic_vertex(local_maxes[idx_max2], smoothed)
                min3_map[xi, yi] = safe_fit_parabolic_vertex(local_mins[idx_min3], smoothed)
                qual_map[xi, yi] = 4 
            elseif !isnothing(idx_max2)
                qual_map[xi, yi] = 3 
            else
                qual_map[xi, yi] = 2 
            end
        catch
            qual_map[xi, yi] = 0
        end
    end

    # -------------------------------------------------------------------------
    # 3. True Spatial RMS and Phase Error Calculation
    # -------------------------------------------------------------------------
    wfe_sq_accumulator = zeros(Float32, 256)
    phase_diff_accumulator = zeros(Float32, 256) 
    wfe_counts = zeros(Int, 256)

    for xi in 1:num_slm_x, yi in 1:num_slm_y
        if qual_map[xi, yi] == 4
            boundaries = [min1_map[xi, yi], max1_map[xi, yi], min2_map[xi, yi], max2_map[xi, yi], min3_map[xi, yi]]
            smoothed_profile = smooth_profile(intensity_cube[xi, yi, :], 2)
            phase_baselines = [0.0f0, Float32(π), Float32(2π), Float32(3π)]
            
            for s in 1:4
                start_idx = round(Int, boundaries[s])
                end_idx   = round(Int, boundaries[s+1])
                
                seg_intensities = smoothed_profile[start_idx:end_idx]
                seg_voltages    = Float32.( (start_idx-1):(end_idx-1) ) # commanded phase steps.
                
                
                i_min, i_max = extrema(seg_intensities)
                norm_i = (i_max > i_min) ? (seg_intensities .- i_min) ./ (i_max - i_min) : zeros(Float32, length(seg_intensities))
                
                local_phi = (s == 1 || s == 3) ? 2.0f0 .* asin.(sqrt.(norm_i)) : 2.0f0 .* acos.(sqrt.(norm_i))
                abs_phi = local_phi .+ phase_baselines[s]
                
                for (idx, v_step) in enumerate(seg_voltages)
                    ideal_phi = v_step * (4.0f0 * Float32(π) / 255.0f0)
                    phase_error_rad = abs_phi[idx] - ideal_phi
                    
                    phase_error_mlambda = (phase_error_rad / (2.0f0 * Float32(π))) * 1000.0f0
                    
                    v_idx = round(Int, v_step) + 1
                    wfe_sq_accumulator[v_idx] += phase_error_mlambda^2
                    phase_diff_accumulator[v_idx] += phase_error_rad 
                    wfe_counts[v_idx] += 1
                end
            end
        end
    end

    rms_wfe_curve = sqrt.(wfe_sq_accumulator ./ max.(wfe_counts, 1))
    avg_phase_diff_curve = phase_diff_accumulator ./ max.(wfe_counts, 1)
    global_rms_val = sqrt(sum(rms_wfe_curve.^2) / 256.0)
    
    println("Independent Validation Sweep Complete. Scalar Global RMS WFE: $(round(global_rms_val, digits=4)) mλ")

    return global_rms_val, rms_wfe_curve, avg_phase_diff_curve 
end

# --- Run sweeps ---
t0 = time()


println("\n>>> Condition 1: Running Custom Regional LUT Sweep [T = 0 Minutes]...")
regional_lut_func(xi, yi, p_idx, v) = regional_lut_matrix[xi, yi, p_idx]
regional_t0_rms, regional_t0_wfe, regional_t0_diff = run_independent_validation_sweep(regional_lut_func)

display(run_figure_E_validation(
    regional_lut_matrix, pixel_quality_map, 
    min1_map, min3_map, affine_matrix,
    slm_x_range, slm_y_range, 
    σ_col_final, σ_row_final, phase_dark
))


const LINEAR_LUT_PATH = "C:\\Users\\nanolab\\Documents\\MeadowLark\\MeadowLark Lut file\\1024x1024_linearVoltage.lut"
const MFR_LUT_PATH    = "C:\\Users\\nanolab\\Documents\\MeadowLark\\MeadowLark Lut file\\slm7831_at633.lut"
num_steps = 256 


println("\n>>> Condition 2: Running Manufacturer Default Linear Sweep...")
Meadowlark.loadlut(MFR_LUT_PATH)
mfg_lut_func(xi, yi, p_idx, v) = v
mfg_rms, mfg_wfe, mfg_diff = run_independent_validation_sweep(mfg_lut_func)
Meadowlark.loadlut(LINEAR_LUT_PATH)

println("\n>>> Condition 3: Running Global Calibrated Sweep (Spatial Average)...")
global_calib_lut = zeros(Float32, num_steps)
for p in 1:num_steps
    valid_pixels = filter(v -> v > 0.0f0 && v <= 255.0f0, regional_lut_matrix[:, :, p])
    global_calib_lut[p] = isempty(valid_pixels) ? 0.0f0 : mean(valid_pixels)
end
global_lut_func(xi, yi, p_idx, v) = global_calib_lut[p_idx]
global_rms, global_wfe, global_diff = run_independent_validation_sweep(global_lut_func)




println("\n>>> Entering 15-Minute Environmental Stability Hold...")
elapsed = time() - t0
remaining = max(0.0, 15*60 - elapsed)

println("\n>>> Waiting $(round(remaining/60, digits=1)) more minutes...")
sleep(remaining)

println("\n>>> Condition 4: Running Custom Regional LUT Sweep [T = 15 Minutes]...")
regional_t15_rms, regional_t15_wfe, regional_t15_diff = run_independent_validation_sweep(regional_lut_func)

println("\n>>> Entering 15-Minute Environmental Stability Hold...")
elapsed = time() - t0
remaining = max(0.0, 30*60 - elapsed)

println("\n>>> Waiting $(round(remaining/60, digits=1)) more minutes...")
sleep(remaining)

println("\n>>> Condition 5: Running Custom Regional LUT Sweep [T = 30 Minutes]...")
regional_t30_rms, regional_t30_wfe, regional_t30_diff = run_independent_validation_sweep(regional_lut_func)

println("\n>>> Entering 15-Minute Environmental Stability Hold...")
elapsed = time() - t0
remaining = max(0.0, 45*60 - elapsed)  

println("\n>>> Waiting $(round(remaining/60, digits=1)) more minutes...")
sleep(remaining)
println("\n>>> Condition 6: Running Custom Regional LUT Sweep [T = 45 Minutes]...")
regional_t45_rms, regional_t45_wfe, regional_t45_diff = run_independent_validation_sweep(regional_lut_func)


# -----------------------------------------------------------------------------
# GRAPHING LOGIC
# -----------------------------------------------------------------------------
phase_axis = (0:255) .* (4.0f0 * Float32(π) / 255.0f0)
valid_mask_4pi = (phase_axis .> 0.2) .& (phase_axis .< (4.0f0 * π - 0.2))

mfr_cutoff_idx = findlast(p -> p <= 2.0f0 * Float32(π), phase_axis)
mfr_phase_axis = phase_axis[1:mfr_cutoff_idx]
valid_mask_2pi = (mfr_phase_axis .> 0.2) .& (mfr_phase_axis .< (2.0f0 * π - 0.2))


# ==============================================================================
# Visualization: Graph 2.F - True RMS Wavefront Error Comparison
# ==============================================================================
fig_2f = Figure(size=(1000, 600))

ax_2f = CM.Axis(
    fig_2f[1, 1],
    title = "Graph 2.F: Spatial RMS Wavefront Error vs. Commanded Phase",
    xlabel = "Commanded Phase (Radians)",
    ylabel = "Wavefront Error W_rms [mλ]",
    yscale = log10, 
    xticks = (0:π:4π, ["0", "1π", "2π", "3π", "4π"])
)

mfr_wfe_truncated = mfg_wfe[1:mfr_cutoff_idx]

lines!(ax_2f, mfr_phase_axis[valid_mask_2pi], mfr_wfe_truncated[valid_mask_2pi], color=:gray, linewidth=2.5, label="1. Manufacturer Default (RMS: $(round(mfg_rms, digits=1)) mλ)")
lines!(ax_2f, phase_axis[valid_mask_4pi], global_wfe[valid_mask_4pi],       color=:darkorange,  linewidth=2.5, label="3. Global Average (RMS: $(round(global_rms, digits=1)) mλ)")
lines!(ax_2f, phase_axis[valid_mask_4pi], regional_t0_wfe[valid_mask_4pi],  color=:forestgreen, linewidth=3.0, label="4. Regional LUT T=0 (RMS: $(round(regional_t0_rms, digits=1)) mλ) ")
lines!(ax_2f, phase_axis[valid_mask_4pi], regional_t15_wfe[valid_mask_4pi],  color=:blue, linewidth=3.0, label="5. Regional LUT T=15 (RMS: $(round(regional_t15_rms, digits=1)) mλ) ")
lines!(ax_2f, phase_axis[valid_mask_4pi], regional_t30_wfe[valid_mask_4pi],  color=:red, linewidth=3.0, label="6. Regional LUT T=30 (RMS: $(round(regional_t30_rms, digits=1)) mλ) ")
lines!(ax_2f, phase_axis[valid_mask_4pi], regional_t45_wfe[valid_mask_4pi],  color=:purple, linewidth=3.0, label="7. Regional LUT T=45 (RMS: $(round(regional_t45_rms, digits=1)) mλ) ")
axislegend(ax_2f, position = :rt, framevisible = true)
display(fig_2f)


# ==============================================================================
# Visualization: Graph 2.H - Phase Discrepancy
# ==============================================================================
fig_2h = Figure(size=(1000, 600))

ax_2h = CM.Axis(
    fig_2h[1, 1],
    title = "Graph 2.H: Phase Discrepancy (Measured - Intended)",
    xlabel = "Commanded Phase (Radians)",
    ylabel = "Delta Phase Δϕ [rad]",
    xticks = (0:π:4π, ["0", "1π", "2π", "3π", "4π"]),
    yticks = (-π/4:π/8:π/4, ["-π/4", "-π/8", "0", "π/8", "π/4"]) 
)

ylims!(ax_2h, -π/4, π/4)
xlims!(ax_2h, 0, 4π)
hlines!(ax_2h, [0.0], color=:black, linestyle=:dash, linewidth=1.5)

#lines!(ax_2h, mfr_phase_axis[valid_mask_2pi], mfg_diff[1:mfr_cutoff_idx][valid_mask_2pi], 
       #color=:gray, linewidth=2.5, label="1. Manufacturer Default")
lines!(ax_2h, phase_axis[valid_mask_4pi], global_diff[valid_mask_4pi], 
       color=:darkorange, linewidth=2.5, label="1. Global Average")
lines!(ax_2h, phase_axis[valid_mask_4pi], regional_t0_diff[valid_mask_4pi], 
       color=:forestgreen, linewidth=3.0, label="2. Regional LUT T=0")
lines!(ax_2h, phase_axis[valid_mask_4pi], regional_t15_diff[valid_mask_4pi], 
       color=:blue, linewidth=3.0, label="3. Regional LUT T=15")
lines!(ax_2h, phase_axis[valid_mask_4pi], regional_t30_diff[valid_mask_4pi], 
       color=:red, linewidth=3.0, label="4. Regional LUT T=30")
lines!(ax_2h, phase_axis[valid_mask_4pi], regional_t45_diff[valid_mask_4pi], 
       color=:purple, linewidth=3.0, label="5. Regional LUT T=45")
axislegend(ax_2h, position = :rt, framevisible = true)
display(fig_2h)




# Visualization: Graph 2.G - Calibrated LUT Heatmaps for Constant Phases
# ==============================================================================
# Define target phases and their string labels
target_phases = [0.0, π, 2π, 3π, 4π]
target_labels = ["0", "π", "2π", "3π", "4π"]

# Convert the phases into the corresponding 1-256 LUT step index
target_indices = [round(Int, (p / (4.0 * π)) * 255) + 1 for p in target_phases]

# Increase height slightly to accommodate the title without overlapping
fig_2g = Figure(size=(1200, 350))

# Put the title explicitly in Row 1, spanning columns 1 through 6
Label(fig_2g[1, 1:6], "Graph 2.G: Calibrated LUT Gray Value Potential", fontsize=20, font=:bold, tellwidth=false)

custom_cmap = cgrad([:darkblue, :deepskyblue, :forestgreen, :darkorange, :yellow])
hmap_ref = nothing 

# Create a universal invalid mask based on the 4π slice (the last slice).
# If a pixel is valid, its 4π gray value will definitely be > 0. 
# This prevents accidentally masking valid 0 gray values in the 0 rad slice.
invalid_mask = regional_lut_matrix[:, :, end] .<= 0.0f0


all_values = Float32[]

for idx in target_indices
    slice = Float32.(regional_lut_matrix[:, :, idx])
    append!(all_values, slice[.!invalid_mask])
end

global_min = minimum(all_values)
global_max = maximum(all_values)



for (i, idx) in enumerate(target_indices)
    # Ensure square pixels and assign to Row 2
    ax_2g = CM.Axis(
        fig_2g[2, i], 
        title = "Phase: $(target_labels[i]) rad",
        aspect = DataAspect() 
    )
    hidedecorations!(ax_2g) 
    
    slice_2d = Float32.(regional_lut_matrix[:, :, idx])

    # Apply the universal mask to set dark spots to NaN
    slice_2d[invalid_mask] .= NaN32
    
    hmap = heatmap!(ax_2g, slice_2d, 
        colormap = custom_cmap, 
        nan_color = :black, 
        colorrange = (global_min, global_max) 
    )
    
    if i == 1
        hmap_ref = hmap
    end
end

# Add the shared colorbar to Row 2, Column 6
Colorbar(fig_2g[2, 6], hmap_ref, label="Gray Value [0 - 255]", height=Relative(0.8))

display(fig_2g)






display(run_figure_E_validation(
    regional_lut_matrix, pixel_quality_map, 
    min1_map, min3_map, affine_matrix,
    slm_x_range, slm_y_range, 
    σ_col_final, σ_row_final, phase_dark
))


 


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
        center_spot_mask[col, row] = fg #(x, y), yes i think i reversed names of col and row...
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