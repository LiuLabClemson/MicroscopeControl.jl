
using FileIO, Images, ImageFiltering
using ImageView
using CairoMakie
CM = CairoMakie
using Statistics

imgpath = "C:\\Users\\shengl\\Downloads\\SLM_Image_Checkerboard_Raw.png"
img = load(imgpath)
gray = Gray.(img)

grid_size = 45
h1 = ones(grid_size,grid_size) # kernel
img_r = imfilter(gray, centered(h1))

localmax = mapwindow(maximum, img_r, (grid_size, grid_size)) .== img_r

img_max = localmax .* img_r
threshold = 10
pts = findall(img_max.>threshold)

# Group by y with ±5 tolerance, then sort x within each y group
y_tolerance = 10
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
fig = Figure(size=(800, 600))
ax = CM.Axis(fig[1, 1], title="grid center detection")
heatmap!(ax, Float64.(gray), colormap=:grays, colorrange=(0, 0.3))
scatter!(ax, x_coords, y_coords, color=:transparent, markersize=10,strokecolor=:red, strokewidth=2)
fig

# Estimate grid size from y-coordinates
y_diff = diff(y_coords)
grid_size_est = mean(y_diff[y_diff.>20]')





using Dates
using JLD2
datadir = "W:\\Projects\\CU-MINFLUX\\SLM calibration\\"
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename = datadir * "Grayscale_scan_exposure_100ms" * timestamp * ".jld2"
@save filename slmstack 
