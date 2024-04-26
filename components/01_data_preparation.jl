### A Pluto.jl notebook ###
# v0.19.40

#> [frontmatter]
#> title = "Data Preparation"
#> description = "Techniques for data loading, augmentation, and dataset splitting."

using Markdown
using InteractiveUtils

# ╔═╡ 6d8f9217-86c2-46cb-876c-ed76b08b2093
using PlutoUI: TableOfContents

# ╔═╡ f3c31a5c-b58d-478f-8514-3275518f7e86
using TestImages: testimage

# ╔═╡ 6b5c8e8c-e26a-462d-a661-322ff817ff45
using ImageSegmentation: felzenszwalb, labels_map

# ╔═╡ c550da11-d789-44cb-a9ef-a67c6b3d3a59
using ImageCore: Gray

# ╔═╡ 613a9a19-c3fe-4aeb-80c8-ff348a827a87
using ImageTransformations: imresize

# ╔═╡ 18cd6ec0-17fd-4fc1-9e70-c24b71f76f68
using ImageFiltering: imfilter, KernelFactors.gaussian

# ╔═╡ 82c15681-146b-4431-9780-11515e4935a0
using CoordinateTransformations: LinearMap, AffineMap

# ╔═╡ b8f56bdd-2ba9-4e61-aa37-2946dcca2127
using ImageTransformations: warp

# ╔═╡ 7d604ad0-f495-4189-9f8c-3f29de66506a
using LinearAlgebra: I

# ╔═╡ 44d1d71c-271a-4e50-ab07-b9778eeb1616
TableOfContents()

# ╔═╡ 2de531b2-a90d-4d38-9c14-602d78e95714
# ╠═╡ show_logs = false
img = testimage("cameraman")

# ╔═╡ 314f7b2e-0c07-499c-8fae-d63b1b834d50
# ╠═╡ show_logs = false
label = Gray.(round.(Int, labels_map(felzenszwalb(img, 1)) .> 1))

# ╔═╡ 844d42a7-173b-4f4c-9d4f-d2b3f475cc36
md"""
# Datasets
Various types of dataset creation examples
"""

# ╔═╡ 5a3c55b6-f9d6-48d8-9da4-e0de7875b2fa
md"""
# Preprocessing
"""

# ╔═╡ b5b8a6b2-bb76-4264-b351-459e10fe69be
md"""
## Cropping
"""

# ╔═╡ ca2eb630-5ffe-470a-80fd-3f8504c159b7
md"""
### `crop`

```julia
crop(img, label=nothing; target_size=(128, 128, 96), max_crop_percentage=0.15)
```

Crops an image and its corresponding label (if provided) to a target size.

#### Arguments
- `img`: The input image.
- `label`: Optional. The corresponding label image. Defaults to `nothing`.
- `target_size`: Optional. The target size of the cropped image and label as a tuple of integers. Defaults to `(128, 128, 96)`.
- `max_crop_percentage`: Optional. The maximum percentage of the image that can be cropped in each dimension. Defaults to `0.15`.

#### Returns
The cropped image and the corresponding cropped label (if provided).
"""

# ╔═╡ 8d70bb6a-5dd2-40c9-8c29-4ac0cfa2dded
function crop(
    img, label=nothing;
    target_size=nothing, max_crop_percentage=0.15)
	
	current_size = size(img)
	N = length(current_size)
	
	if isnothing(target_size)
		target_size = current_size
	elseif length(target_size) != N
		throw(ArgumentError("The dimensionality of target_size must match the dimensionality of the input image."))
	end
	
	# Calculate the maximum allowable crop size, only for dimensions larger than target size
	max_crop_size = [min(cs, round(Int, ts + (cs - ts) * max_crop_percentage)) for (cs, ts) in zip(current_size, target_size)]
	
	# Center crop for dimensions needing cropping
	start_idx = [
		cs > ts ? max(1, div(cs, 2) - div(ms, 2)) : 1 for (cs, ts, ms) in zip(current_size, target_size, max_crop_size)
	]
	end_idx = start_idx .+ max_crop_size .- 1
	
	# Create a tuple of ranges for indexing
	crop_ranges = tuple([start_idx[i]:end_idx[i] for i in 1:N]...)
	
	cropped_img = img[crop_ranges...]
	cropped_label = isnothing(label) ? nothing : label[crop_ranges...]
	
	return cropped_img, cropped_label
end

# ╔═╡ d5d2694d-9f50-455b-b3a0-a95f50dae02f
md"""
#### Example
"""

# ╔═╡ 3652d276-2f0f-4849-891b-88081f80c3ca
img_crop, label_crop = crop(img, label; target_size=(100, 400));

# ╔═╡ 5c79d869-b61f-438a-89c8-d535b084ddc8
img_crop

# ╔═╡ 9f97d611-716b-484d-b7ce-2e13d14d92fc
label_crop

# ╔═╡ 66a7d160-418f-4647-a849-116427797c3a
md"""
## Resizing
"""

# ╔═╡ 299fd6b1-d582-4bf8-a05a-a268f79baa51
md"""
### `resize`

```julia
resize(img, label=nothing; target_size=(128, 128, 96))
```

Resizes an image and its corresponding label (if provided) to a target size.

#### Arguments
- `img`: The input image.
- `label`: Optional. The corresponding label image. Defaults to `nothing`.
- `target_size`: Optional. The target size of the resized image and label as a tuple of integers. Defaults to `(128, 128, 96)`.

#### Returns
The resized image and the corresponding resized label (if provided).
"""

# ╔═╡ acc86ef3-a463-443b-94c0-b61f0d3827cb
function resize(
	img, label=nothing;
	target_size=(128, 128, 96))
	
    resized_img = imresize(img, target_size)
    resized_label = isnothing(label) ? nothing : imresize(label, target_size)
    return resized_img, resized_label
end

# ╔═╡ 15ee4795-a37f-4661-ad08-f37e321ac177
md"""
#### Example
"""

# ╔═╡ 44d256a0-dad7-4c9c-80a2-5aefa2726d1f
img_resize, label_resize = resize(img, label; target_size=(64, 64));

# ╔═╡ 7533e898-e6bf-4d42-adf8-cfd26a0626ee
size(img), size(label)

# ╔═╡ cc4bf46f-e244-4614-9994-2a1909581b0c
size(img_resize), size(label_resize)

# ╔═╡ ca581faf-2bfb-45b9-a856-0f44f1a6286d
img_resize

# ╔═╡ d4f54fed-3d2e-4cb9-b3e4-67dcf3b15235
label_resize

# ╔═╡ 28c004a8-97dc-4793-bd04-22fd94f5a825
md"""
## Encoding
"""

# ╔═╡ b3082a2f-da0b-4fc0-aeff-2ae6cf062646
md"""
### `one_hot_encode`

```julia
one_hot_encode(img, label; num_classes=2)
```

Performs one-hot encoding on a label image.

#### Arguments
- `img`: The input image.
- `label`: The corresponding label image.
- `num_classes`: Optional. The number of classes in the label image. Defaults to `2`.

#### Returns
The input image reshaped to include a channel dimension and the one-hot encoded label.
"""

# ╔═╡ edcb7d8a-bdf5-435e-ba8d-76303f53cc42
function one_hot_encode(
	img, label;
	num_classes=2)

    img_one_hot = reshape(img, size(img)..., 1)

    if isnothing(label)
        return img_one_hot, nothing
    else
        label_one_hot = Float32.(zeros(size(label)..., num_classes))

		if ndims(label) == 2
	        for k in 1:num_classes
	            label_one_hot[:, :, k] = Float32.(label .== (k-1))
	        end
		elseif ndims(label) == 3
			for k in 1:num_classes
	            label_one_hot[:, :, :, k] = Float32.(label .== (k-1))
	        end
		end

        return img_one_hot, label_one_hot
    end
end

# ╔═╡ ad847d40-3223-4bd5-9097-e283c5a564d1
md"""
#### Example
"""

# ╔═╡ f44679d6-9f61-4d9f-aa11-b4ca0998a9fa
img_one_hot, label_one_hot = one_hot_encode(img_resize, label_resize);

# ╔═╡ a44a7324-4341-4e52-8356-20c2532afba9
size(img_one_hot), size(label_one_hot)

# ╔═╡ 5d596eef-4441-4f0d-aae1-4c3c4808e8f2
img_one_hot[:, :, 1]

# ╔═╡ 172c36ef-1465-448c-8097-ec18a956bfda
Gray.(label_one_hot[:, :, 1])

# ╔═╡ 2ee6c0e9-122f-4aaf-af84-c18faa565681
md"""
# Augmentation
[*Credit: Augmentor.jl*](https://github.com/Evizero/Augmentor.jl/tree/master)
"""

# ╔═╡ ca543a27-c824-4474-83ad-d1008a1886e0
md"""
## Blurring
"""

# ╔═╡ ede7b17d-dba0-485d-a230-bc859af0b8d2
md"""
### `rand_gaussian_blur`

```julia
rand_gaussian_blur(img, label=nothing; p=0.5, k=3, σ=0.3 * ((k - 1) / 2 - 1) + 0.8)
```

Blurs an image using a Gaussian filter with a given probability.

#### Arguments
- `img`: The input image.
- `label`: Optional. The corresponding label image. Defaults to `nothing`.
- `p`: Optional. The probability of applying the blur. Defaults to 0.5.
- `k`: Optional. `Integer` or `AbstractVector` of `Integer` that denote the kernel size. It must be an odd positive number. Defaults to 3.
- `σ`: Optional. `Real` or `AbstractVector` of `Real` that denote the standard deviation. It must be a positive number.
       Defaults to `0.3 * ((k - 1) / 2 - 1) + 0.8`.

#### Returns
The blurred image and the corresponding label if the blur is applied, otherwise the original image and label.
"""

# ╔═╡ 2f1850cd-6c51-49b3-93ab-a9bdcf221185
function rand_gaussian_blur(img, label=nothing; p=0.5, k=3, σ=0.3 * ((k - 1) / 2 - 1) + 0.8)
	if rand() < p
		if isa(k, Integer)
			k = fill(k, ndims(img))
		end
		if isa(σ, Real)
			σ = fill(σ, ndims(img))
		end

		minimum(k) > 0 || throw(ArgumentError("Kernel size must be positive: $(k)"))
		minimum(σ) > 0 || throw(ArgumentError("σ must be positive: $(σ)"))

		kernel = gaussian(σ, k)
		blurred_img = imfilter(img, kernel)
		blurred_label = isnothing(label) ? nothing : round.(Int, imfilter(Float32.(label), kernel))
		return blurred_img, blurred_label
	else
		return img, label
	end
end

# ╔═╡ ab0403f9-0571-45ee-99a6-65283be2b1e7
md"""
#### Example
"""

# ╔═╡ 443bad85-27fb-4ab9-935f-30895e98e247
blurred_img, blurred_label = rand_gaussian_blur(img, label; p=1.0, k=101);

# ╔═╡ 41814925-edc8-41ce-b51f-a059ee415309
# ╠═╡ show_logs = false
blurred_img

# ╔═╡ 8f57cab3-a3f8-43e4-8e55-1bcb5e78b997
# ╠═╡ show_logs = false
Gray.(blurred_label)

# ╔═╡ c430ef24-b1a0-40d5-981e-6686ec05b74e
md"""
## Flipping
"""

# ╔═╡ 7063648e-cff5-4169-8498-bd5ce5b013d0
md"""
### `rand_flip_x`

```julia
rand_flip_x(img, label=nothing; p=0.5)
```

Flips an image horizontally (along the y-axis) with a given probability.

#### Arguments
- `img`: The input image.
- `label`: Optional. The corresponding label image. Defaults to `nothing`.
- `p`: Optional. The probability of applying the flip. Defaults to 0.5.

#### Returns
The flipped image and the corresponding label if the flip is applied, otherwise the original image and label.
"""

# ╔═╡ 3aa5e73e-2943-439e-aaaf-46c83b48e4ce
function rand_flip_x(img, label=nothing; p=0.5)
    if rand() < p
        flipped_img = reverse(img; dims=2)
        flipped_label = isnothing(label) ? nothing : reverse(label; dims=2)
        return flipped_img, flipped_label
    else
        return img, label
    end
end

# ╔═╡ afb778f1-a259-441d-9b93-eb93273e1afe
md"""
#### Example
"""

# ╔═╡ 397a94fb-d30d-45bf-9d48-f5c61d5781f4
flipped_img_x, flipped_label_x = rand_flip_x(img, label; p=1.0);

# ╔═╡ 0c1c287f-4ec1-484d-b195-d86461807558
# ╠═╡ show_logs = false
flipped_img_x

# ╔═╡ 663efd3c-f54c-48bd-bc47-ad0668d6819d
# ╠═╡ show_logs = false
Gray.(flipped_label_x)

# ╔═╡ d88ff44c-1de1-4f9d-a78f-23437a1716df
md"""
### `rand_flip_y`

```julia
rand_flip_y(img, label=nothing; p=0.5)
```

Flips an image vertically (along the x-axis) with a given probability.

#### Arguments
- `img`: The input image.
- `label`: Optional. The corresponding label image. Defaults to `nothing`.
- `p`: Optional. The probability of applying the flip. Defaults to 0.5.

#### Returns
The flipped image and the corresponding label if the flip is applied, otherwise the original image and label.
"""

# ╔═╡ 090ad379-8478-48ee-bf15-3325fe90e48a
function rand_flip_y(img, label=nothing; p=0.5)
    if rand() < p
        flipped_img = reverse(img; dims=1)
        flipped_label = isnothing(label) ? nothing : reverse(label; dims=1)
        return flipped_img, flipped_label
    else
        return img, label
    end
end

# ╔═╡ 57bf48f3-0a6e-4ecd-bd01-240c608a3c75
md"""
#### Example
"""

# ╔═╡ a8c6d3e9-cbfc-404f-b776-cf0123ae97dd
flipped_img_y, flipped_label_y = rand_flip_y(img, label; p=1.0);

# ╔═╡ c15718dc-3776-463c-9a81-9e4e9c22eecc
# ╠═╡ show_logs = false
flipped_img_y

# ╔═╡ f4fa0c09-c3d7-4278-bd7e-6f7460196d99
# ╠═╡ show_logs = false
Gray.(flipped_label_y)

# ╔═╡ 00a29af3-c9ac-4dcd-b663-2d1eaca3798e
md"""
## Shearing
"""

# ╔═╡ a42693f1-ac74-4c65-81f7-cc4eb044ced2
md"""
### `rand_shear_x`

```julia
rand_shear_x(img, label=nothing; p=0.5, degree=-10:10)
```

Shears an image horizontally by a randomly selected angle in degrees with a given probability.

#### Arguments
- `img`: The input image.
- `label`: Optional. The corresponding label image. Defaults to `nothing`.
- `p`: Optional. The probability of applying the shear. Defaults to 0.5.
- `degree`: Optional. `Real` or `AbstractVector` of `Real` that denote the shearing angle(s) in degrees.
            If a vector is provided, then a random element will be sampled. Defaults to -10:10.

#### Returns
The sheared image and the corresponding label if the shear is applied, otherwise the original image and label.
"""

# ╔═╡ 240faa96-c7a5-49d6-b370-544435042526
md"""
!!! warning
    Needs to be adjusted for n-dimensional data. Only works with 2D
"""

# ╔═╡ 8cb91f8f-cc04-42eb-9591-e25e35aa5712
function _shear(img::AbstractArray, angle::Real, axis::Int)
    N = ndims(img)
    shear_matrix = Matrix{Float64}(I, N, N)
    for i in 1:N
        if i != axis
            shear_matrix[i, axis] = tan(deg2rad(-angle))
        end
    end
    tform = AffineMap(LinearMap(shear_matrix), zeros(N))
    return warp(img, tform)
end

# ╔═╡ 6c988769-a55c-4bf5-aa1d-a2220943e252
function rand_shear_x(img, label=nothing; p=0.5, degree=-10:10)
	if rand() < p
		length(degree) > 0 || throw(ArgumentError("The number of different angles passed to \"shear_x(...)\" must be non-zero"))
		(minimum(degree) >= -70 && maximum(degree) <= 70) || throw(ArgumentError("The specified shearing angle(s) must be in the interval [-70, 70]"))
	
		angle = rand(degree)
		sheared_img = _shear(img, angle, 1)
		sheared_label = isnothing(label) ? nothing : round.(Bool, _shear(label, angle, 1))
		return sheared_img, sheared_label
	else
		return img, label
	end
end

# ╔═╡ c3b9a1bf-0a93-4884-a7c6-27bc50c0389d
md"""
#### Example
"""

# ╔═╡ 849a3e64-31e0-4bec-8e16-c8f2a999dc1d
sheared_img_x, sheared_label_x = rand_shear_x(img, label; p = 1.0, degree = -30:30);

# ╔═╡ 1c3b51f9-e125-4886-bf64-332ec01d0f57
# ╠═╡ show_logs = false
sheared_img_x

# ╔═╡ 22d6c5d2-6526-4bff-afc1-63ece2ea0970
# ╠═╡ show_logs = false
Gray.(sheared_label_x)

# ╔═╡ d97a7e15-a528-437d-ac7a-900d3236572d
md"""
### `rand_shear_y`

```julia
rand_shear_y(img, label=nothing; p=0.5, degree=-10:10)
```

Shears an image vertically by a randomly selected angle in degrees with a given probability.

#### Arguments
- `img`: The input image.
- `label`: Optional. The corresponding label image. Defaults to `nothing`.
- `p`: Optional. The probability of applying the shear. Defaults to 0.5.
- `degree`: Optional. `Real` or `AbstractVector` of `Real` that denote the shearing angle(s) in degrees.
            If a vector is provided, then a random element will be sampled. Defaults to -10:10.

#### Returns
The sheared image and the corresponding label if the shear is applied, otherwise the original image and label.
"""

# ╔═╡ 2e7ee006-129f-4506-ac36-b46e77cd4682
function rand_shear_y(img, label=nothing; p=0.5, degree=-10:10)
    if rand() < p
        length(degree) > 0 || throw(ArgumentError("The number of different angles passed to \"shear_y(...)\" must be non-zero"))
        (minimum(degree) >= -70 && maximum(degree) <= 70) || throw(ArgumentError("The specified shearing angle(s) must be in the interval [-70, 70]"))

        angle = rand(degree)
        sheared_img = _shear(img, angle, 2)
        sheared_label = isnothing(label) ? nothing : round.(Bool, _shear(label, angle, 2))
        return sheared_img, sheared_label
    else
        return img, label
    end
end

# ╔═╡ e4d09f77-cab6-447c-a90a-28f22415e564
md"""
#### Example
"""

# ╔═╡ 64be76b2-3110-41f9-838c-5f431592022d
sheared_img_y, sheared_label_y = rand_shear_y(img, label; p = 1);

# ╔═╡ 8f9094d3-1273-4f88-b09b-1fa35d73bf68
# ╠═╡ show_logs = false
sheared_img_y

# ╔═╡ 62cfe4c1-f998-484e-934b-1297e9f51371
Gray.(sheared_label_y)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CoordinateTransformations = "150eb455-5306-5404-9cee-2592286d6298"
ImageCore = "a09fc81d-aa75-5fe9-8630-4744c3626534"
ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
ImageSegmentation = "80713f31-8817-5129-9cf8-209ff8fb23e1"
ImageTransformations = "02fcd773-0e25-5acc-982a-7f6622650795"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
CoordinateTransformations = "~0.6.3"
ImageCore = "~0.10.2"
ImageFiltering = "~0.7.8"
ImageSegmentation = "~1.8.2"
ImageTransformations = "~0.10.1"
PlutoUI = "~0.7.59"
TestImages = "~1.8.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "99f1c94a1564d4c55ad004c4bfa176a04e5df4a1"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "133a240faec6e074e07c31ee75619c90544179cf"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.10.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "575cd02e080939a33b6df6c5853d14924c08e35b"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.23.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "3863330da5466410782f2bffc64f3d505a6a8334"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.10.0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "8e2eae13d144d545ef829324f1f0a5a4fe4340f3"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.3.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "8d2e786fd090199a91ecbf4a66d03aedd0fb24d4"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.11+4"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "3ff0ca203501c3eedde3c6fa7fd76b703c336b5f"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.2"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e0884bdf01bbbb111aea77c348368a86fb4b5ab6"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.1"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be50fe8df3acbffa0274a744f1a99d29c45a57f4"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "5ea6acdd53a51d897672edb694e3cc2912f3f8a7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.46"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "a13f3be5d84b9c95465d743c82af0b094ef9c2e2"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.169"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "80b2833b56d466b3858d565adcd16a4a05f2089b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.1.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded64ff6d4fdd1cb68dfcbb818c69e144a5b2e4c"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.16"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "e64b4f5ea6b7389f6f046d13d4896a8f9c1ba71e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "763a8ceb07833dd51bb9e3bbca372de32c0605ad"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "2a0a5d8569f481ff8840e3b7c84bbf188db6a3fe"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.0"

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

    [deps.Rotations.weakdeps]
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d2fdac9ff3906e27f7a618d47b676941baa6c80c"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.10"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "5d66818a39bb04bf328e92bc933ec5b4ee88e436"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StringDistances]]
deps = ["Distances", "StatsAPI"]
git-tree-sha1 = "5b2ca70b099f91e54d98064d5caf5cc9b541ad06"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.11.3"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TestImages]]
deps = ["AxisArrays", "ColorTypes", "FileIO", "ImageIO", "ImageMagick", "OffsetArrays", "Pkg", "StringDistances"]
git-tree-sha1 = "0567860ec35a94c087bd98f35de1dddf482d7c67"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.8.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "ac377f0a248753a1b1d58bbc92a64f5a726dfb71"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.66"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═6d8f9217-86c2-46cb-876c-ed76b08b2093
# ╠═f3c31a5c-b58d-478f-8514-3275518f7e86
# ╠═6b5c8e8c-e26a-462d-a661-322ff817ff45
# ╠═c550da11-d789-44cb-a9ef-a67c6b3d3a59
# ╠═44d1d71c-271a-4e50-ab07-b9778eeb1616
# ╠═2de531b2-a90d-4d38-9c14-602d78e95714
# ╠═314f7b2e-0c07-499c-8fae-d63b1b834d50
# ╟─844d42a7-173b-4f4c-9d4f-d2b3f475cc36
# ╟─5a3c55b6-f9d6-48d8-9da4-e0de7875b2fa
# ╟─b5b8a6b2-bb76-4264-b351-459e10fe69be
# ╟─ca2eb630-5ffe-470a-80fd-3f8504c159b7
# ╠═8d70bb6a-5dd2-40c9-8c29-4ac0cfa2dded
# ╟─d5d2694d-9f50-455b-b3a0-a95f50dae02f
# ╠═3652d276-2f0f-4849-891b-88081f80c3ca
# ╠═5c79d869-b61f-438a-89c8-d535b084ddc8
# ╠═9f97d611-716b-484d-b7ce-2e13d14d92fc
# ╟─66a7d160-418f-4647-a849-116427797c3a
# ╟─299fd6b1-d582-4bf8-a05a-a268f79baa51
# ╠═613a9a19-c3fe-4aeb-80c8-ff348a827a87
# ╠═acc86ef3-a463-443b-94c0-b61f0d3827cb
# ╟─15ee4795-a37f-4661-ad08-f37e321ac177
# ╠═44d256a0-dad7-4c9c-80a2-5aefa2726d1f
# ╠═7533e898-e6bf-4d42-adf8-cfd26a0626ee
# ╠═cc4bf46f-e244-4614-9994-2a1909581b0c
# ╠═ca581faf-2bfb-45b9-a856-0f44f1a6286d
# ╠═d4f54fed-3d2e-4cb9-b3e4-67dcf3b15235
# ╟─28c004a8-97dc-4793-bd04-22fd94f5a825
# ╟─b3082a2f-da0b-4fc0-aeff-2ae6cf062646
# ╠═edcb7d8a-bdf5-435e-ba8d-76303f53cc42
# ╟─ad847d40-3223-4bd5-9097-e283c5a564d1
# ╠═f44679d6-9f61-4d9f-aa11-b4ca0998a9fa
# ╠═a44a7324-4341-4e52-8356-20c2532afba9
# ╠═5d596eef-4441-4f0d-aae1-4c3c4808e8f2
# ╠═172c36ef-1465-448c-8097-ec18a956bfda
# ╟─2ee6c0e9-122f-4aaf-af84-c18faa565681
# ╟─ca543a27-c824-4474-83ad-d1008a1886e0
# ╟─ede7b17d-dba0-485d-a230-bc859af0b8d2
# ╠═18cd6ec0-17fd-4fc1-9e70-c24b71f76f68
# ╠═2f1850cd-6c51-49b3-93ab-a9bdcf221185
# ╟─ab0403f9-0571-45ee-99a6-65283be2b1e7
# ╠═443bad85-27fb-4ab9-935f-30895e98e247
# ╠═41814925-edc8-41ce-b51f-a059ee415309
# ╠═8f57cab3-a3f8-43e4-8e55-1bcb5e78b997
# ╟─c430ef24-b1a0-40d5-981e-6686ec05b74e
# ╟─7063648e-cff5-4169-8498-bd5ce5b013d0
# ╠═3aa5e73e-2943-439e-aaaf-46c83b48e4ce
# ╟─afb778f1-a259-441d-9b93-eb93273e1afe
# ╠═397a94fb-d30d-45bf-9d48-f5c61d5781f4
# ╠═0c1c287f-4ec1-484d-b195-d86461807558
# ╠═663efd3c-f54c-48bd-bc47-ad0668d6819d
# ╟─d88ff44c-1de1-4f9d-a78f-23437a1716df
# ╠═090ad379-8478-48ee-bf15-3325fe90e48a
# ╟─57bf48f3-0a6e-4ecd-bd01-240c608a3c75
# ╠═a8c6d3e9-cbfc-404f-b776-cf0123ae97dd
# ╠═c15718dc-3776-463c-9a81-9e4e9c22eecc
# ╠═f4fa0c09-c3d7-4278-bd7e-6f7460196d99
# ╟─00a29af3-c9ac-4dcd-b663-2d1eaca3798e
# ╟─a42693f1-ac74-4c65-81f7-cc4eb044ced2
# ╠═82c15681-146b-4431-9780-11515e4935a0
# ╠═b8f56bdd-2ba9-4e61-aa37-2946dcca2127
# ╠═7d604ad0-f495-4189-9f8c-3f29de66506a
# ╟─240faa96-c7a5-49d6-b370-544435042526
# ╠═8cb91f8f-cc04-42eb-9591-e25e35aa5712
# ╠═6c988769-a55c-4bf5-aa1d-a2220943e252
# ╟─c3b9a1bf-0a93-4884-a7c6-27bc50c0389d
# ╠═849a3e64-31e0-4bec-8e16-c8f2a999dc1d
# ╠═1c3b51f9-e125-4886-bf64-332ec01d0f57
# ╠═22d6c5d2-6526-4bff-afc1-63ece2ea0970
# ╟─d97a7e15-a528-437d-ac7a-900d3236572d
# ╠═2e7ee006-129f-4506-ac36-b46e77cd4682
# ╟─e4d09f77-cab6-447c-a90a-28f22415e564
# ╠═64be76b2-3110-41f9-838c-5f431592022d
# ╠═8f9094d3-1273-4f88-b09b-1fa35d73bf68
# ╠═62cfe4c1-f998-484e-934b-1297e9f51371
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
