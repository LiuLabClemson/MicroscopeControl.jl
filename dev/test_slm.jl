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
using JLD2



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
# I have three ideas to do this. Method (A), compute the camera coordiante that corresponds to the slm pixel, 
# then compute the average intensity of the 2 by 2 grid. The second method, (B), would compute the affine polygon 
# corresponding to the corners of the SLM pixel. Then only use pixels whose center is in the polygon. 
# My third method (C) is to use bilinear interpolation with the 4 closest pixels. This would give the intensity at the transformed
# point as opposed to the integral of the intensity over the footprint; however, this should be a good first order approximation
# that should give a good smooth curve. 
# I have been assuming SLM coordiantes refer to the center of the given SLM pixel, hopefully that is true...
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



#---------------------------------------------------------------------------
#More diagonitic figures, dont normallly run this time for diffraction limit
#not working....
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
        smoothed_profile = smooth_profile(raw_profile, 2) # Matches Step 8 filter
        
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
#10.  Use the inverse interferometric relationship to transform the normalized intensities into 
#phase values stepping up by $\pi$ radians per segment.
# Fig 2.c
# ==============================================================================



# ==============================================================================
#11. Invert to create the Look-Up Table
# Fig 2.d
# ==============================================================================




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
cx = 512#514
cy = 512#561
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