# Examples Assets (User-provided)

This directory stores sample input images and optional masks for trying the CLI examples in [examples/README.md](../README.md). No binary images are included in the repository.

Add your own files:
- Input images: JPEG (.jpg, .jpeg) or PNG (.png)
- Optional mask: PNG (.png), same width/height as the input image

Recommendations:
- Use images with the short side around 512–1024 px for faster runs and good quality
- Keep file sizes modest (typically < 4–8 MB)
- For masks, prefer a binary black/white or grayscale image with the same dimensions as the input

Suggested filenames used by the examples:
- Input image: ./examples/assets/input.jpg
- Mask image: ./examples/assets/mask.png

Tip: avoid committing large binaries. If needed, you may add patterns to [.gitignore](../../.gitignore) (for example):
```gitignore
/examples/assets/*.jpg
/examples/assets/*.jpeg
/examples/assets/*.png
```

After placing your files here, run the example commands from the project root as shown in [examples/README.md](../README.md).