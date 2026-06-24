
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







h1 = ones(2,2)
img_r = imfilter(gray, centered(h1))









img_edge = detect_edges(img_r, Canny(spatial_scale = 4.5, 
    low = ImageEdgeDetection.Percentile(80), 
    high = ImageEdgeDetection.Percentile(90)))





function nonmax_suppression(x::Matrix{T}) where {T}
    I, J = size(x)
    y = copy(x)
    for j = 1:J
        for i = 1:I
            c = x[i,j]
            for j2 = max(j - 1, 1):min(j + 1, J),
                i2 = max(i - 1, 1):min(i + 1, I)
                if x[i2, j2] > c
                    y[i, j] = 0
                    break
                end
            end
        end
    end
    return y
end

x = imfilter(img_edge, KernelFactors.gaussian((1.2, 1.2)))
g1, g2 = imgradients(x,KernelFactors.sobel, "reflect")
α1 = g1 .* g1 - g2 .* g2
α2 = 2 * g1 .* g2
α1 = imfilter(α1, KernelFactors.gaussian((1.2, 1.2)))
α2 = imfilter(α2, KernelFactors.gaussian((1.2, 1.2)))
h1 = [(i^2 - j^2) / max(1, i^2 + j^2) for i = -15:15 , j = -15:15]
h2 = [2 * i * j / max(1, i^2 + j^2) for i = -15:15 , j = -15:15]
r = imfilter(α2, centered(h2)) - imfilter(α1, centered(h1))
r = nonmax_suppression(r)

# Create RGB image with r .> 2 in red and x in grayscale
r_mask = Float32.(r.>2)
x_normalized = Float32.(img_r ./ maximum(img_r))  # Normalize gray to [0, 1]
rgb_img = RGB.(r_mask, x_normalized, x_normalized)
imshow(rgb_img)




h1 = ones(45,45)
img_r = imfilter(gray, centered(h1))

localmax = mapwindow(maximum, img_r, (45, 45)) .== img_r

pts = findall(localmax)