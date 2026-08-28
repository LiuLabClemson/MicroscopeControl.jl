using Distributions
using Optim
using CairoMakie
CM = CairoMakie

# generate 2D gaussian blob, the model function
function genblob(θ,sz)
    y, x,σy,σx,photons, bg = θ
    py = Normal(y,σy)
    px = Normal(x,σx)

    
    #roi = [(y,x) for y in 1:sz, x in 1:sz]
    blob = [pdf(py,y)*pdf(px,x)*photons + bg for y in 1:sz, x in 1:sz]

    return blob
end

# add poisson noise to the image
function addnoise(imgin)
    noisy_image = zeros(size(imgin))
    
    for j in axes(imgin,2)
        for i in axes(imgin,1)
            noisy_image[i,j] = rand(Poisson(imgin[i,j]))
        end
    end

    return noisy_image
end

# define the loss function, negative log likelihood
function loss(θ,data,sz)
    model = genblob(θ,sz)
    out = 0.0
    for idx in eachindex(model)
        if model[idx] > 0
            out += data[idx]*log(model[idx]) - model[idx]
        end
    end
    #LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))
    return -out
end


# test 2D gaussian fitting
# simulate data
sz = 17
photons = 200
bg = 2
σx = 2.0
σy = 1.0
θ = [sz/2,sz/2,σy,σx,photons,bg]

psf = genblob(θ,sz)
data = addnoise(psf)

# estimate parameters
θ_0 = [sz/2,sz/2,1.1,1.1,sum(data),bg]
fobjective = x -> loss(x,data,sz)
out=optimize(fobjective,θ_0,NelderMead())
θ_est = out.minimizer

# display results
psf_fit = genblob(θ_est,sz)
println(θ)
println(θ_est)



# Create figure
fig = Figure(size = (900, 300))
ax1 = CM.Axis(fig[1, 1], title = "Data")
heatmap!(ax1, data)
ax2 = CM.Axis(fig[1, 2], title = "PSF Fit")
heatmap!(ax2, psf_fit)
ax3 = CM.Axis(fig[1, 3], title = "PSF")
heatmap!(ax3, psf)
# clean axes
for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
    hidespines!(ax)
end

display(fig)

# optional, save figure    
#resultdir = "W:\\Projects\\CU-MINFLUX\\SLM calibration\\results\\"
#savename = "blobfit_test"
#save(resultdir*savename*".png", fig, px_per_unit = 2)
