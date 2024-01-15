### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "Image Segmentation"
#> description = "Guide on 3D heart segmentation in CT images."

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d4f7e164-f9a6-47ee-85a7-dd4e0dec10ee
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".."); Pkg.instantiate()

# ╔═╡ 8d4a6d5a-c437-43bb-a3db-ab961b218c2e
using PlutoUI: TableOfContents, Slider, bind

# ╔═╡ 83b95cee-90ed-4522-b9a8-79c082fce02e
using Random: default_rng, seed!

# ╔═╡ 7353b7ce-8b33-4602-aed7-2aa24864aca5
using HTTP: download

# ╔═╡ de5efc37-db19-440e-9487-9a7bea84996d
using Tar: extract

# ╔═╡ 3ab44a2a-692f-4603-a5a8-81f1d260c13e
using MLUtils: DataLoader, splitobs, mapobs, getobs

# ╔═╡ 562b3772-89cc-4390-87c3-e7260c8aa86b
using NIfTI: niread

# ╔═╡ da9cada1-7ea0-4b6b-a338-d8e08b668d28
using ImageTransformations: imresize

# ╔═╡ db2ccf3a-437a-4dfa-ad05-2526c0e2bde0
using Glob: glob

# ╔═╡ 317c1571-d232-4cab-ac10-9fc3b7ad33b0
# ╠═╡ show_logs = false
using LuxCUDA

# ╔═╡ 8e2f2c6d-127d-42a6-9906-970c09a22e61
using CairoMakie: Figure, Axis, heatmap!, scatterlines!, axislegend, ylims!

# ╔═╡ a3f44d7c-efa3-41d0-9509-b099ab7f09d4
using Lux

# ╔═╡ a6669580-de24-4111-a7cb-26d3e727a12e
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ dfc9377a-7cc1-43ba-bb43-683d24e67d79
using ComputerVisionMetrics: hausdorff_metric, dice_metric

# ╔═╡ c283f9a3-6a76-4186-859f-21cd9efc131f
using ChainRulesCore: ignore_derivatives

# ╔═╡ 70bc36db-9ee3-4e1d-992d-abbf55c52070
using Losers: hausdorff_loss, dice_loss

# ╔═╡ 2f6f0755-d71f-4239-a72b-88a545ba8ca1
using Dates: now

# ╔═╡ b04c696b-b404-4976-bfc1-51889ef1d60f
using JLD2: jldsave

# ╔═╡ e457a411-2e7b-43b3-a247-23eff94222b0
using DataFrames: DataFrame

# ╔═╡ ec8d4131-d8c0-4bdc-9479-d96dc712567c
# ╠═╡ show_logs = false
using ParameterSchedulers: Exp

# ╔═╡ c8d6553a-90df-4aeb-aa6d-a213e16fab48
TableOfContents()

# ╔═╡ af50e5f3-1a1c-47e5-a461-ffbee0329309
begin
    rng = default_rng()
    seed!(rng, 0)
end

# ╔═╡ cdfd2412-897d-4642-bb69-f8031c418446
function download_dataset(heart_url, target_directory)
    if isempty(readdir(target_directory))
        local_tar_file = joinpath(target_directory, "heart_dataset.tar")
		download(heart_url, "heart_dataset.tar")
		extract("heart_dataset.tar", target_directory)
		data_dir = joinpath(target_directory, readdir(target_directory)...)
        return data_dir
    else
        @warn "Target directory is not empty. Aborting download and extraction."
        return taget_directory
    end
end

# ╔═╡ b1516500-ad83-41d2-8a1d-093cd0d948e3
heart_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar"

# ╔═╡ 3e896957-61d8-4750-89bd-be02383417ec
target_directory = mktempdir()

# ╔═╡ 99211382-7de9-4e97-872f-d0c01b8f8307
# ╠═╡ show_logs = false
data_dir = download_dataset(heart_url, target_directory)

# ╔═╡ 6d34b756-4da8-427c-91f5-dfb022c4e715
begin
	struct HeartSegmentationDataset
		image_paths::Vector{String}
		label_paths::Vector{String}
	end
	
	function HeartSegmentationDataset(root_dir::String; is_test::Bool = false)
	    if is_test
	        image_paths = glob("*.nii*", joinpath(root_dir, "imagesTs"))
	        return HeartSegmentationDataset(image_paths, String[])
	    else
	        image_paths = glob("*.nii*", joinpath(root_dir, "imagesTr"))
	        label_paths = glob("*.nii*", joinpath(root_dir, "labelsTr"))
	        return HeartSegmentationDataset(image_paths, label_paths)
	    end
	end
	
	Base.length(d::HeartSegmentationDataset) = length(d.image_paths)
	
	function Base.getindex(d::HeartSegmentationDataset, i::Int)
	    image = niread(d.image_paths[i]).raw
	    if !isempty(d.label_paths)
	        label = niread(d.label_paths[i]).raw
	        return (image, label)
	    else
	        return image
	    end
	end
	
	function Base.getindex(d::HeartSegmentationDataset, idxs::AbstractVector{Int})
	    images = Vector{Array{Float32, 3}}(undef, length(idxs))
		labels = isempty(d.label_paths) ? nothing : Vector{Array{UInt8, 3}}(undef, length(idxs))
	
	    for (index, i) in enumerate(idxs)
	        images[index] = niread(d.image_paths[i]).raw
	        if labels !== nothing
	            labels[index] = niread(d.label_paths[i]).raw
	        end
	    end
	
	    if labels !== nothing
	        return (images, labels)
	    else
	        return images
	    end
	end
end

# ╔═╡ 65dac38d-f955-4058-b577-827d7f8b3db4
md"""
# Setup
"""

# ╔═╡ 7cf78ac3-cedd-479d-bc50-769f7b772060
md"""
## Environment
"""

# ╔═╡ af798f6b-7549-4253-b02b-2ed20dc1125b
md"""
## Randomness
"""

# ╔═╡ f0e64ba5-5e11-4ddb-91d3-2a34c60dc6bf
md"""
# Data Preparation
"""

# ╔═╡ ec7734c3-33a5-43c7-82db-2db4dbdc9587
md"""
## Dataset
"""

# ╔═╡ 9577b91b-faa4-4fc5-9ec2-ed8ca94f2afe
data = HeartSegmentationDataset(data_dir)

# ╔═╡ ae3d24e4-2216-4744-9093-0d2a8bbaae2d
md"""
## Preprocessing
"""

# ╔═╡ 61bbecff-868a-4ca0-afb9-dc8aff786783
function crop_image(
    volume::Array{T, 3}, 
    target_size::Tuple{Int, Int, Int}, 
    max_crop_percentage::Float64 = 0.15
) where T
    current_size = size(volume)

	# Calculate the maximum allowable crop size, only for dimensions larger than target size
	max_crop_size = [min(cs, round(Int, ts + (cs - ts) * max_crop_percentage)) for (cs, ts) in zip(current_size, target_size)]

    # Center crop for dimensions needing cropping
	start_idx = [
		cs > ts ? max(1, div(cs, 2) - div(ms, 2)) : 1 for (cs, ts, ms) in zip(current_size, target_size, max_crop_size)
	]
    end_idx = start_idx .+ max_crop_size .- 1
	cropped_volume = volume[start_idx[1]:end_idx[1], start_idx[2]:end_idx[2], start_idx[3]:end_idx[3]]

    return cropped_volume
end

# ╔═╡ 6b8e2236-cd17-452f-8e68-93c9418027cd
function resize_image(
    volume::Array{T, 3}, 
    target_size::Tuple{Int, Int, Int}
) where T
    return imresize(volume, target_size)
end

# ╔═╡ 0a03b692-045f-4321-8065-ebca13e94a96
function one_hot_encode(label::Array{T, 3}, num_classes::Int) where T
	one_hot = zeros(T, size(label)..., num_classes)
	
    for k in 1:num_classes
        one_hot[:, :, :, k] = label .== k-1
    end
	
    return one_hot
end

# ╔═╡ e91fa0c9-cde9-4416-9e6a-3faa4f8af717
function preprocess_data(pair, target_size)
    # Check if pair[1] and pair[2] are individual arrays or collections of arrays
    is_individual = ndims(pair[1]) == 3 && ndims(pair[2]) == 3

    if is_individual
        # Handle a single pair
        resized_image = resize_image(pair[1], target_size)
		processed_image = Float32.(reshape(resized_image, size(resized_image)..., 1))

        resized_label = resize_image(pair[2], target_size)
        one_hot_label = Float32.(one_hot_encode(resized_label, 2))

        return (processed_image, one_hot_label)
    else
        # Handle a batch of pairs
        resized_images = [resize_image(img, target_size) for img in pair[1]]
		processed_images = [Float32.(reshape(img, size(img)..., 1)) for img in resized_images]

        resized_labels = [resize_image(lbl, target_size) for lbl in pair[2]]
		one_hot_labels = [Float32.(one_hot_encode(lbl, 2)) for lbl in resized_labels]

        return (processed_images, one_hot_labels)
    end
end

# ╔═╡ 217d073d-c145-4b3d-85c4-eee8d22d1018
if LuxCUDA.functional()
	# target_size = (256, 256, 128)
	target_size = (128, 128, 96)
else
	target_size = (64, 64, 32)
end

# ╔═╡ f2b8a5ae-1c5c-47ba-8215-8ef7c5619d68
transformed_data = mapobs(
	x -> preprocess_data(x, target_size),
	data
)

# ╔═╡ 03bab55a-6e5e-4b9f-b56a-7e9f993576eb
md"""
## Dataloaders
"""

# ╔═╡ d40f19dc-f06e-44ef-b82b-9763ff1f1189
train_data, val_data = splitobs(transformed_data; at = 0.75)

# ╔═╡ 4d75f114-225f-45e2-a683-e82ff137d909
bs = 4

# ╔═╡ 2032b7e6-ceb7-4c08-9b0d-bc704f5e4104
begin
	train_loader = DataLoader(train_data; batchsize = bs, collate = true)
	val_loader = DataLoader(val_data; batchsize = bs, collate = true)
end

# ╔═╡ 2ec43028-c1ab-4df7-9cfe-cc1a4919a7cf
md"""
# Data Visualization
"""

# ╔═╡ a6316144-c809-4d2a-bda1-d5128dcf89d3
md"""
## Original Data
"""

# ╔═╡ f8fc2cee-c1bd-477d-9595-9427e8764bd6
image_raw, label_raw = getobs(data, 1);

# ╔═╡ 7cb986f8-b338-4046-b569-493e443a8dcb
@bind z1 Slider(axes(image_raw, 3), show_value = true, default = div(size(image_raw, 3), 2))

# ╔═╡ d7e75a72-8281-432c-abab-c254f8c94d3c
let
	f = Figure(size = (700, 500))
	ax = Axis(
		f[1, 1],
		title = "Original Image"
	)
	heatmap!(image_raw[:, :, z1]; colormap = :grays)

	ax = Axis(
		f[1, 2],
		title = "Original Label (Overlayed)"
	)
	heatmap!(image_raw[:, :, z1]; colormap = :grays)
	heatmap!(label_raw[:, :, z1]; colormap = (:jet, 0.4))
	f
end

# ╔═╡ 9dc89870-3d99-472e-8974-712e34a3a789
md"""
## Transformed Data
"""

# ╔═╡ 0f5d7796-2c3d-4b74-86c1-a1d4e3922011
image_tfm, label_tfm = getobs(transformed_data, 1);

# ╔═╡ 51e9e7d9-a1d2-4fd1-bdad-52851d9498a6
typeof(image_tfm), typeof(label_tfm)

# ╔═╡ 803d918a-66ce-4ed3-a33f-5dda2dd7288e
unique(label_tfm)

# ╔═╡ 6e2bfcfb-77e3-4532-a14d-10f4b91f2f54
@bind z2 Slider(1:target_size[3], show_value = true, default = div(target_size[3], 2))

# ╔═╡ bae79c05-034a-4c39-801a-01229b618e94
let
	f = Figure(size = (700, 500))
	ax = Axis(
		f[1, 1],
		title = "Transformed Image"
	)
	heatmap!(image_tfm[:, :, z2, 1]; colormap = :grays)

	ax = Axis(
		f[1, 2],
		title = "Transformed Label (Overlayed)"
	)
	heatmap!(image_tfm[:, :, z2, 1]; colormap = :grays)
	heatmap!(label_tfm[:, :, z2, 2]; colormap = (:jet, 0.4))
	f
end

# ╔═╡ 95ad5275-63ca-4f2a-9f3e-6c6a340f5cd4
md"""
# Model
"""

# ╔═╡ 773aace6-14ad-46f6-a1a6-692247231e90
md"""
## Helper Blocks
"""

# ╔═╡ 1588d84a-c5f7-4be6-9295-c3594d77b08f
function conv_layer(
	k, in_channels, out_channels;
	pad=2, stride=1, activation=relu)
	
    return Chain(
        Conv((k, k, k), in_channels => out_channels, pad=pad, stride=stride),
        BatchNorm(out_channels),
        WrappedFunction(activation)
    )
end

# ╔═╡ f9b0aa7f-d660-4d6f-bd5d-721e5c809b13
function contract_block(
	in_channels, mid_channels, out_channels;
	k=5, stride=2, activation=relu)
	
    return Chain(
        conv_layer(k, in_channels, mid_channels),
        conv_layer(k, mid_channels, out_channels),
        Chain(
            Conv((2, 2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ e682f461-43d7-492a-85a9-2d46e829a125
function expand_block(
	in_channels, mid_channels, out_channels;
	k=5, stride=2, activation=relu)
	
    return Chain(
        conv_layer(k, in_channels, mid_channels),
        conv_layer(k, mid_channels, out_channels),
        Chain(
            ConvTranspose((2, 2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ 36ad66d6-c484-4073-bf01-1f7ec7012373
md"""
## Unet
"""

# ╔═╡ f55e3c0f-6abe-423c-8319-96146f30eebd
function Unet(in_channels::Int = 1, out_channels::Int = in_channels)
    return Chain(
        # Initial Convolution Layer
        conv_layer(5, in_channels, 4),

        # Contracting Path
        contract_block(4, 8, 8),
        contract_block(8, 16, 16),
        contract_block(16, 32, 32),
        contract_block(32, 64, 64),

        # Bottleneck Layer
        conv_layer(5, 64, 128),

        # Expanding Path
        expand_block(128, 64, 64),
        expand_block(64, 32, 32),
        expand_block(32, 16, 16),
        expand_block(16, 8, 8),

        # Final Convolution Layer
        Conv((1, 1, 1), 8 => out_channels)
    )
end

# ╔═╡ bbdaf5c5-9faa-4b61-afab-c0242b8ca034
model = Unet(1, 2)

# ╔═╡ df2dd9a7-045c-44a5-a62c-8d9f2541dc14
md"""
# Training Set Up
"""

# ╔═╡ 1b5ae165-1069-4638-829a-471b907cce86
import CSV

# ╔═╡ 69880e6d-162a-4aae-94eb-103bd35ac3c9
import Zygote

# ╔═╡ 12d42392-ad7b-4c5f-baf5-1f2c6052669e
import Optimisers

# ╔═╡ 7cde37c8-4c59-4583-8995-2b01eda95cb3
md"""
## Optimiser
"""

# ╔═╡ ca87af51-1e56-48f7-8343-1b4e8fe1c91a
begin
    struct Scheduler{T, F}<: Optimisers.AbstractRule
        constructor::F
        schedule::T
    end

    _get_opt(scheduler::Scheduler, t) = scheduler.constructor(scheduler.schedule(t))

    Optimisers.init(o::Scheduler, x::AbstractArray) =
        (t = 1, opt = Optimisers.init(_get_opt(o, 1), x))

    function Optimisers.apply!(o::Scheduler, state, x, dx)
        opt = _get_opt(o, state.t)
        new_state, new_dx = Optimisers.apply!(opt, state.opt, x, dx)

        return (t = state.t + 1, opt = new_state), new_dx
    end
end

# ╔═╡ 0390bcf5-4cd6-49ba-860a-6f94f8ba6ded
function create_optimiser(ps)
    opt = Scheduler(Exp(λ = 1e-2, γ = 0.8)) do lr Optimisers.Adam(0.1f0) end
    return Optimisers.setup(opt, ps)
end

# ╔═╡ a25bdfe6-b24d-446b-926f-6e0727d647a2
md"""
## Loss function
"""

# ╔═╡ 08f2911c-90e7-418e-b9f2-a0722a857bf1
function compute_loss(x, y, model, ps, st)
    # Get model predictions
	y_pred, st = Lux.apply(model, x, ps, st)

    # Apply sigmoid activation
    y_pred_sigmoid = sigmoid.(y_pred)

    # Compute loss
    loss = 0.0
    for b in axes(y, 5)  # Iterate over the batch dimension
        _y_pred = y_pred_sigmoid[:, :, :, 2, b]
        _y = y[:, :, :, 2, b]

        dsc = dice_loss(_y_pred, _y)  # Use the adjusted dice_loss function
        loss += dsc
    end

    # Average the loss over the batch
    return loss / size(y, 5), y_pred_sigmoid, st
end

# ╔═╡ 45949f7f-4e4a-4857-af43-ff013dbdd137
md"""
# Training
"""

# ╔═╡ 402ba194-350e-4ff3-832b-6651be1d9ce7
dev = gpu_device()

# ╔═╡ 6ec3e34b-1c57-4cfb-a50d-ee786c2e4559
begin
	ps, st = Lux.setup(rng, model)
	ps, st = ps |> dev, st |> dev
end

# ╔═╡ b7561ff5-d704-4301-b038-c02bbba91ae2
md"""
## Train
"""

# ╔═╡ 1e79232f-bda2-459a-bc03-85cd8afab3bf
function train_model(model, ps, st, train_loader, val_loader, num_epochs, dev)

    opt_state = create_optimiser(ps)

    # Initialize DataFrame to store metrics
    metrics_df = DataFrame(
        "Epoch" => Int[], 
		"Train_Loss" => Float64[],
        "Validation_Loss" => Float64[], 
        "Dice_Metric" => Float64[],
        "Hausdorff_Metric" => Float64[],
        "Epoch_Duration" => String[]
    )

    for epoch in 1:num_epochs
        @info "Epoch: $epoch"

        # Start timing the epoch
        epoch_start_time = now()

		# Training Phase
		num_batches_train = 0
		total_loss = 0.0
        for (x, y) in train_loader
			num_batches_train += 1
			@info "Step: $num_batches_train"
			x, y = x |> dev, y |> dev

			(loss, y_pred, st), back = Zygote.pullback(compute_loss, x, y, model, ps, st)
			total_loss += loss
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

        end

		# Calculate and log time taken for the epoch
        epoch_duration = now() - epoch_start_time
		
		avg_train_loss = total_loss / num_batches_train
		@info "avg_train_loss: $avg_train_loss"

		# Validation Phase
		val_loss = 0.0
		total_dice = 0.0
		total_hausdorff = 0.0
		num_batches = 0
		num_images = 0
		st_ = Lux.testmode(st)
		for (x, y) in val_loader
			num_batches += 1
		    x, y = x |> dev, y |> dev
		    (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
		    val_loss += loss
			
		 #    # Process batch for metrics
			# y_pred_cpu, y_cpu = y_pred |> cpu_device(), y |> cpu_device()
		 #    for b in axes(y_cpu, 5)
			# 	num_images += 1
				
		 #        _y_pred = Bool.(round.(sigmoid.(y_pred_cpu[:, :, :, 1, b])))
		 #        _y = Bool.(y_cpu[:, :, :, 1, b])
		
		 #        total_dice += dice_metric(_y_pred, _y)
		 #        total_hausdorff += hausdorff_metric(_y_pred, _y)
		 #    end
		    
		end
		
		# Calculate average metrics
		avg_val_loss = val_loss / num_batches
		# avg_dice = total_dice / num_images
		# avg_hausdorff = total_hausdorff / num_images
		# @info "avg_val_loss: $avg_val_loss"
		# @info "avg_dice: $avg_dice"
		# @info "avg_hausdorff: $avg_hausdorff"

		# # Save new model if dice or hausdorff metrics improve
		# if epoch > 1 && (avg_dice > metrics_df[end, :Dice_Metric] || avg_hausdorff < metrics_df[end, :Hausdorff_Metric])
		# 	jldsave("params_img_seg_epoch_$epoch.jld2"; ps)
		# 	jldsave("states_img_seg_epoch_$epoch.jld2"; st)
		# end
		
        # Append metrics to the DataFrame
		# push!(metrics_df, [epoch, avg_train_loss, avg_val_loss, avg_dice, avg_hausdorff, string(epoch_duration)])
		push!(metrics_df, [epoch, avg_train_loss, avg_val_loss, 0, 0, string(epoch_duration)])

        # Write DataFrame to CSV file
        CSV.write("img_seg_metrics.csv", metrics_df)

        @info "Metrics logged for Epoch $epoch"
    end

    return ps, st
end

# ╔═╡ a2e88851-227a-4719-8828-6064f9d3ef81
if LuxCUDA.functional()
	num_epochs = 30
else
	num_epochs = 10
end

# ╔═╡ 5cae73af-471c-4068-b9ff-5bc03dd0472d
# ╠═╡ disabled = true
#=╠═╡
ps_final, st_final = train_model(model, ps, st, train_loader, val_loader, num_epochs, dev);
  ╠═╡ =#

# ╔═╡ 0dee7c0e-c239-49a4-93c9-5a856b3da883
md"""
## Visualize Training
"""

# ╔═╡ 0bf3a26a-9e18-43d0-b059-d37e8f2e3645
df = CSV.read("img_seg_metrics.csv", DataFrame)

# ╔═╡ bc72bff8-a4a8-4736-9aa2-0e87eed243ba
let
	f = Figure()
	ax = Axis(
		f[1, 1],
		title = "Losses"
	)
	
	scatterlines!(df[!, :Epoch], df[!, :Train_Loss], label = "Train Loss")
	scatterlines!(df[!, :Epoch], df[!, :Validation_Loss], label = "Validation Loss")

	ylims!(low = 0, high = 1.2)
	axislegend(ax; position = :lc)

	ax = Axis(
		f[2, 1],
		title = "Metrics"
	)
	scatterlines!(df[!, :Epoch], df[!, :Dice_Metric], label = "Dice Metric")
	scatterlines!(df[!, :Epoch], df[!, :Hausdorff_Metric], label = "Hausdorff Metric")

	axislegend(ax; position = :lc)

	
	f
end

# ╔═╡ ca57dee1-2669-4202-801d-c88b4d3d7c8d
md"""
## Save Final Model
"""

# ╔═╡ 7b9b554e-2999-4c57-805e-7bc0d7a0b4e7
#=╠═╡
jldsave("params_img_seg_final.jld2"; ps_final)
  ╠═╡ =#

# ╔═╡ 6432d227-3ff6-4230-9f52-c3e57ba78618
#=╠═╡
jldsave("states_img_seg_final.jld2"; st_final)
  ╠═╡ =#

# ╔═╡ 33b4df0d-86e0-4728-a3bc-928c4dff1400
md"""
# Model Inference
"""

# ╔═╡ edddcb37-ac27-4c6a-a98e-c34525cce108
md"""
## Load Test Images
"""

# ╔═╡ 7c821e74-cab5-4e5b-92bc-0e8f76d36556
test_data = HeartSegmentationDataset(data_dir; is_test = true)

# ╔═╡ 6dafe561-411a-45b9-b0ee-d385136e1568
function preprocess_test_data(image, target_size)
    resized_image = resize_image(image, target_size)
    processed_image = Float32.(reshape(resized_image, size(resized_image)..., 1))
    return processed_image
end

# ╔═╡ fe2cfe67-9d87-4eb7-a3d6-13402afbb99a
transformed_test_data = mapobs(
    x -> preprocess_test_data(x, target_size),
    test_data
)

# ╔═╡ bf325c7f-d43a-4a02-b339-2a84eac1c4ff
test_loader = DataLoader(transformed_test_data; batchsize = 10, collate = true)

# ╔═╡ b206b46a-4261-4727-a4d6-23a305382374
md"""
## Load Best Model
"""

# ╔═╡ 27360e10-ad7e-4fdc-95c5-fef0c5b550dd
md"""
## Predict
"""

# ╔═╡ 3545de13-f283-4431-81e7-3abfa14774de
md"""
## Visualize
"""

# ╔═╡ 13303866-8a40-4325-9334-6de60a2068cd
image_test = getobs(transformed_test_data, 1);

# ╔═╡ 1adace71-2b22-461e-86c5-fe42f7b69958
# typeof(image_test)

# ╔═╡ 24fa3061-6d6b-4efe-a537-6cd6eaa9b045
@bind z3 Slider(axes(image_test, 3), show_value = true, default = div(size(image_test, 3), 2))

# ╔═╡ 2c63c5ff-f364-4f78-bd3c-ac89f32d7b0f
let
	f = Figure(size = (700, 500))
	ax = Axis(
		f[1, 1],
		title = "Test Image"
	)
	heatmap!(image_test[:, :, z3, 1]; colormap = :grays)
	f
end

# ╔═╡ Cell order:
# ╟─65dac38d-f955-4058-b577-827d7f8b3db4
# ╟─7cf78ac3-cedd-479d-bc50-769f7b772060
# ╠═d4f7e164-f9a6-47ee-85a7-dd4e0dec10ee
# ╠═8d4a6d5a-c437-43bb-a3db-ab961b218c2e
# ╠═c8d6553a-90df-4aeb-aa6d-a213e16fab48
# ╟─af798f6b-7549-4253-b02b-2ed20dc1125b
# ╠═83b95cee-90ed-4522-b9a8-79c082fce02e
# ╠═af50e5f3-1a1c-47e5-a461-ffbee0329309
# ╟─f0e64ba5-5e11-4ddb-91d3-2a34c60dc6bf
# ╠═7353b7ce-8b33-4602-aed7-2aa24864aca5
# ╠═de5efc37-db19-440e-9487-9a7bea84996d
# ╠═3ab44a2a-692f-4603-a5a8-81f1d260c13e
# ╠═562b3772-89cc-4390-87c3-e7260c8aa86b
# ╠═da9cada1-7ea0-4b6b-a338-d8e08b668d28
# ╠═db2ccf3a-437a-4dfa-ad05-2526c0e2bde0
# ╟─ec7734c3-33a5-43c7-82db-2db4dbdc9587
# ╠═cdfd2412-897d-4642-bb69-f8031c418446
# ╠═b1516500-ad83-41d2-8a1d-093cd0d948e3
# ╠═3e896957-61d8-4750-89bd-be02383417ec
# ╠═99211382-7de9-4e97-872f-d0c01b8f8307
# ╠═6d34b756-4da8-427c-91f5-dfb022c4e715
# ╠═9577b91b-faa4-4fc5-9ec2-ed8ca94f2afe
# ╟─ae3d24e4-2216-4744-9093-0d2a8bbaae2d
# ╠═61bbecff-868a-4ca0-afb9-dc8aff786783
# ╠═6b8e2236-cd17-452f-8e68-93c9418027cd
# ╠═0a03b692-045f-4321-8065-ebca13e94a96
# ╠═e91fa0c9-cde9-4416-9e6a-3faa4f8af717
# ╠═317c1571-d232-4cab-ac10-9fc3b7ad33b0
# ╠═217d073d-c145-4b3d-85c4-eee8d22d1018
# ╠═f2b8a5ae-1c5c-47ba-8215-8ef7c5619d68
# ╟─03bab55a-6e5e-4b9f-b56a-7e9f993576eb
# ╠═d40f19dc-f06e-44ef-b82b-9763ff1f1189
# ╠═4d75f114-225f-45e2-a683-e82ff137d909
# ╠═2032b7e6-ceb7-4c08-9b0d-bc704f5e4104
# ╟─2ec43028-c1ab-4df7-9cfe-cc1a4919a7cf
# ╠═8e2f2c6d-127d-42a6-9906-970c09a22e61
# ╟─a6316144-c809-4d2a-bda1-d5128dcf89d3
# ╠═f8fc2cee-c1bd-477d-9595-9427e8764bd6
# ╟─7cb986f8-b338-4046-b569-493e443a8dcb
# ╟─d7e75a72-8281-432c-abab-c254f8c94d3c
# ╟─9dc89870-3d99-472e-8974-712e34a3a789
# ╠═0f5d7796-2c3d-4b74-86c1-a1d4e3922011
# ╠═51e9e7d9-a1d2-4fd1-bdad-52851d9498a6
# ╠═803d918a-66ce-4ed3-a33f-5dda2dd7288e
# ╟─6e2bfcfb-77e3-4532-a14d-10f4b91f2f54
# ╟─bae79c05-034a-4c39-801a-01229b618e94
# ╟─95ad5275-63ca-4f2a-9f3e-6c6a340f5cd4
# ╠═a3f44d7c-efa3-41d0-9509-b099ab7f09d4
# ╟─773aace6-14ad-46f6-a1a6-692247231e90
# ╠═1588d84a-c5f7-4be6-9295-c3594d77b08f
# ╠═f9b0aa7f-d660-4d6f-bd5d-721e5c809b13
# ╠═e682f461-43d7-492a-85a9-2d46e829a125
# ╟─36ad66d6-c484-4073-bf01-1f7ec7012373
# ╠═f55e3c0f-6abe-423c-8319-96146f30eebd
# ╠═bbdaf5c5-9faa-4b61-afab-c0242b8ca034
# ╟─df2dd9a7-045c-44a5-a62c-8d9f2541dc14
# ╠═a6669580-de24-4111-a7cb-26d3e727a12e
# ╠═dfc9377a-7cc1-43ba-bb43-683d24e67d79
# ╠═c283f9a3-6a76-4186-859f-21cd9efc131f
# ╠═70bc36db-9ee3-4e1d-992d-abbf55c52070
# ╠═2f6f0755-d71f-4239-a72b-88a545ba8ca1
# ╠═b04c696b-b404-4976-bfc1-51889ef1d60f
# ╠═e457a411-2e7b-43b3-a247-23eff94222b0
# ╠═1b5ae165-1069-4638-829a-471b907cce86
# ╠═69880e6d-162a-4aae-94eb-103bd35ac3c9
# ╠═12d42392-ad7b-4c5f-baf5-1f2c6052669e
# ╠═ec8d4131-d8c0-4bdc-9479-d96dc712567c
# ╟─7cde37c8-4c59-4583-8995-2b01eda95cb3
# ╠═ca87af51-1e56-48f7-8343-1b4e8fe1c91a
# ╠═0390bcf5-4cd6-49ba-860a-6f94f8ba6ded
# ╟─a25bdfe6-b24d-446b-926f-6e0727d647a2
# ╠═08f2911c-90e7-418e-b9f2-a0722a857bf1
# ╟─45949f7f-4e4a-4857-af43-ff013dbdd137
# ╠═402ba194-350e-4ff3-832b-6651be1d9ce7
# ╠═6ec3e34b-1c57-4cfb-a50d-ee786c2e4559
# ╟─b7561ff5-d704-4301-b038-c02bbba91ae2
# ╠═1e79232f-bda2-459a-bc03-85cd8afab3bf
# ╠═a2e88851-227a-4719-8828-6064f9d3ef81
# ╠═5cae73af-471c-4068-b9ff-5bc03dd0472d
# ╟─0dee7c0e-c239-49a4-93c9-5a856b3da883
# ╠═0bf3a26a-9e18-43d0-b059-d37e8f2e3645
# ╟─bc72bff8-a4a8-4736-9aa2-0e87eed243ba
# ╟─ca57dee1-2669-4202-801d-c88b4d3d7c8d
# ╠═7b9b554e-2999-4c57-805e-7bc0d7a0b4e7
# ╠═6432d227-3ff6-4230-9f52-c3e57ba78618
# ╟─33b4df0d-86e0-4728-a3bc-928c4dff1400
# ╟─edddcb37-ac27-4c6a-a98e-c34525cce108
# ╠═7c821e74-cab5-4e5b-92bc-0e8f76d36556
# ╠═6dafe561-411a-45b9-b0ee-d385136e1568
# ╠═fe2cfe67-9d87-4eb7-a3d6-13402afbb99a
# ╠═bf325c7f-d43a-4a02-b339-2a84eac1c4ff
# ╟─b206b46a-4261-4727-a4d6-23a305382374
# ╟─27360e10-ad7e-4fdc-95c5-fef0c5b550dd
# ╟─3545de13-f283-4431-81e7-3abfa14774de
# ╠═13303866-8a40-4325-9334-6de60a2068cd
# ╠═1adace71-2b22-461e-86c5-fe42f7b69958
# ╟─24fa3061-6d6b-4efe-a537-6cd6eaa9b045
# ╟─2c63c5ff-f364-4f78-bd3c-ac89f32d7b0f
