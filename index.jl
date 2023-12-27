### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "Computer Vision Tutorials"
#> tags = ["sidebar"]
#> sidebar = "false"

using Markdown
using InteractiveUtils

# ╔═╡ 4b704955-ea7f-48fc-b756-f2c0d860d60d
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".")

# ╔═╡ bbae4dce-5490-4d2f-a97a-f67c6b93a7cc
using Lux

# ╔═╡ dd52d41f-433e-491b-a89c-158becd790ae
using PlutoUI: TableOfContents

# ╔═╡ 0bdf5876-5998-42fd-ae94-98faa4cff3b2
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ be4d8910-f6c8-421e-99cd-5c188a298ebb
using Losers: dice_loss, hausdorff_loss

# ╔═╡ 257cbe31-6122-4988-b37e-f57260ebc5ea
using ComputerVisionMetrics: dice_metric, hausdorff_metric

# ╔═╡ ee1e03f4-7013-4300-8cd4-1a47272befac
using HTMLStrings: to_html, head, link, script, divv, h1, img, p, span, a, figure, hr

# ╔═╡ ba623422-a111-11ee-070a-3fad79960969
md"""
# Motivation

This repository pulls together packages in the Julia ecosystem focused on various aspects of deep learning for medical imaging and provides a comprehensive overview of how each package might fit together in a cohesive way for training a neural network. Key packages are 

1. DistanceTransforms.jl
2. Losers.jl
3. ComputerVisionMetrics.jl
"""

# ╔═╡ 9cfc3b4d-3ba6-4fb7-ac4b-d85f5358a015
md"""
# QuickStart
"""

# ╔═╡ 14c89d42-0996-4ca9-9d65-8743656e9ffd
TableOfContents()

# ╔═╡ 32cd7420-ad98-4f8b-908b-ec79e1bc39e7
md"""
## Prepare Data
"""

# ╔═╡ 664a5447-3e3f-4569-b864-1f08ecfa4d67
md"""
## Loss Function(s), Optimizers, Etc.
"""

# ╔═╡ e845d847-a924-4f8b-a82f-57227ea2ae6b
md"""
## Model
"""

# ╔═╡ dfb180c6-901f-422d-b4ca-6d0b8604f5f6
md"""
## Training
"""

# ╔═╡ 473e5069-93de-4880-a450-7de2ed8e3949
md"""
## Results
"""

# ╔═╡ c7ef4fe4-a868-4969-834a-677cafbfbaf5
to_html(
	divv(
		divv(:class => "min-h-screen"),
		hr(:class => "h-full")
	)
)

# ╔═╡ 43f3c01f-6e32-4144-a122-488129d4374b
md"""
# Deep Dive
"""

# ╔═╡ 02a475d1-9208-4fb4-b321-bce622e8448d
md"""
## Tutorials
"""

# ╔═╡ f5ca1509-dbf2-40ed-87f6-9ca9ed2e5aba
md"""
## Core Packages
"""

# ╔═╡ dcf15c2f-7356-4e43-8a16-339151ae833b
to_html(hr())

# ╔═╡ d5b8279a-9df1-41a7-a9af-100a7fb8b44c
data_theme = "dracula";

# ╔═╡ 4f46e4eb-1640-4c0d-9867-e2002ecb6529
function index_title_card(title::String, subtitle::String, image_url::String; data_theme::String = "pastel", border_color::String = "primary")
	return to_html(
	    divv(
	        head(
				link(:href => "https://cdn.jsdelivr.net/npm/daisyui@3.7.4/dist/full.css", :rel => "stylesheet", :type => "text/css"),
	            script(:src => "https://cdn.tailwindcss.com")
	        ),
			divv(:data_theme => "$data_theme", :class => "card card-bordered flex justify-center items-center border-$border_color text-center w-full dark:text-[#e6e6e6]",
				divv(:class => "card-body flex flex-col justify-center items-center",
					img(:src => "$image_url", :class => "h-24 w-24 md:h-40 md:w-40 rounded-md", :alt => "$title Logo"),
					divv(:class => "text-5xl font-bold bg-gradient-to-r from-accent to-primary inline-block text-transparent bg-clip-text py-10", "$title"),
					p(:class => "card-text text-md font-serif", "$subtitle"
					)
				)
			)
	    )
	)
end;

# ╔═╡ 3f662845-48cb-477e-a3d9-f1f9dfe7f2d3
index_title_card(
	"ComputerVisionTutorials.jl",
	"Practical tutorials for deep learning in the field of computer vision with an emphasis on medical imaging, using Julia.",
	"https://img.freepik.com/free-photo/modern-hospital-machinery-illuminates-blue-mri-scanner-generated-by-ai_188544-44420.jpg?ga=GA1.1.1694943658.1700350224&semt=ais_ai_generated";
	data_theme = data_theme
)

# ╔═╡ 20770955-46bd-4d07-b759-f08604583b01
struct Article
	title::String
	path::String
	image_url::String
end

# ╔═╡ aa48d79f-00dc-4ab0-bba3-bbefc58c2707
article_list_tutorials = Article[
	Article("Data Preparation", "tutorials/data_preparation.jl", "https://img.freepik.com/premium-photo/stickers-basic-gift-box-open-with-simple-geometric-creative-concept-boxes-gift-design_655090-499328.jpg?ga=GA1.1.1694943658.1700350224&semt=ais_ai_generated"),	
	Article("Models", "tutorials/models.jl", "https://img.freepik.com/premium-photo/pile-lego-bricks-with-word-lego-it_822916-171.jpg?ga=GA1.1.1694943658.1700350224&semt=ais_ai_generated"),
];

# ╔═╡ 49dd6dbd-04dd-4c67-86ec-1111f7ab6ebb
article_list_packages = Article[
	Article("Losers.jl", "Losers.jl/index.jl", "https://img.freepik.com/free-vector/low-self-esteem-woman-looking-into-mirror_23-2148714425.jpg?size=626&ext=jpg&ga=GA1.1.1427368820.1695503713&semt=ais"),	
	Article("DistanceTransforms.jl", "DistanceTransforms.jl/index.jl", "https://img.freepik.com/free-vector/global-communication-background-business-network-vector-design_53876-151122.jpg"),
	Article("ComputerVisionMetrics.jl", "ComputerVisionMetrics.jl/index.jl", "https://img.freepik.com/free-photo/colorful-bar-graph-sits-table-with-dark-background_1340-34474.jpg"),	
];

# ╔═╡ 53bb4fb7-54d9-4626-a388-356db88c738e
function article_card(article::Article, color::String; data_theme = "pastel")
    a(:href => article.path, :class => "w-1/2 p-2",
		divv(:data_theme => "$data_theme", :class => "card card-bordered border-$color text-center dark:text-[#e6e6e6]",
			divv(:class => "card-body justify-center items-center",
				p(:class => "card-title", article.title),
				p("Click to open the notebook")
			),
			figure(
				img(:class =>"w-full h-60", :src => article.image_url, :alt => article.title)
			)
        )
    )
end;

# ╔═╡ 24c3e2cf-bcbb-41b2-a608-801344e455f6
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [article_card(article, "accent"; data_theme = data_theme) for article in article_list_tutorials]...
    )
)

# ╔═╡ 16099593-958f-4670-851b-86e71f3fc1d0
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [article_card(article, "accent"; data_theme = data_theme) for article in article_list_packages]...
    )
)

# ╔═╡ Cell order:
# ╟─3f662845-48cb-477e-a3d9-f1f9dfe7f2d3
# ╟─ba623422-a111-11ee-070a-3fad79960969
# ╟─9cfc3b4d-3ba6-4fb7-ac4b-d85f5358a015
# ╠═4b704955-ea7f-48fc-b756-f2c0d860d60d
# ╠═bbae4dce-5490-4d2f-a97a-f67c6b93a7cc
# ╠═dd52d41f-433e-491b-a89c-158becd790ae
# ╠═0bdf5876-5998-42fd-ae94-98faa4cff3b2
# ╠═be4d8910-f6c8-421e-99cd-5c188a298ebb
# ╠═257cbe31-6122-4988-b37e-f57260ebc5ea
# ╠═14c89d42-0996-4ca9-9d65-8743656e9ffd
# ╟─32cd7420-ad98-4f8b-908b-ec79e1bc39e7
# ╟─664a5447-3e3f-4569-b864-1f08ecfa4d67
# ╟─e845d847-a924-4f8b-a82f-57227ea2ae6b
# ╟─dfb180c6-901f-422d-b4ca-6d0b8604f5f6
# ╟─473e5069-93de-4880-a450-7de2ed8e3949
# ╟─c7ef4fe4-a868-4969-834a-677cafbfbaf5
# ╟─43f3c01f-6e32-4144-a122-488129d4374b
# ╟─02a475d1-9208-4fb4-b321-bce622e8448d
# ╟─24c3e2cf-bcbb-41b2-a608-801344e455f6
# ╟─f5ca1509-dbf2-40ed-87f6-9ca9ed2e5aba
# ╟─16099593-958f-4670-851b-86e71f3fc1d0
# ╟─dcf15c2f-7356-4e43-8a16-339151ae833b
# ╟─ee1e03f4-7013-4300-8cd4-1a47272befac
# ╟─d5b8279a-9df1-41a7-a9af-100a7fb8b44c
# ╟─4f46e4eb-1640-4c0d-9867-e2002ecb6529
# ╟─20770955-46bd-4d07-b759-f08604583b01
# ╟─aa48d79f-00dc-4ab0-bba3-bbefc58c2707
# ╟─49dd6dbd-04dd-4c67-86ec-1111f7ab6ebb
# ╟─53bb4fb7-54d9-4626-a388-356db88c738e
