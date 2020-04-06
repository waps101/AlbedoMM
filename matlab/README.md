# Matlab code

In building the albedo morphable model, we developed some matlab code that extends the [MatlabRenderer](https://github.com/waps101/MatlabRenderer).

## Image sampling and Poisson blending

The functions **MR_sample_image** and **MR_poisson_blending** can be used to sample images onto a mesh and then blend them into a seamless, per-vertex texture. We include some sample data and a demo script that show how to do this (**demo_blending**).

## Ambient occlusion

We experimented with removing ambient occlusion from light stage diffuse albedo maps. In the end we didn't use these versions when building model but we could not find any matlab ambient occlusion code and thought this might be useful to someone. It's very slow but does work. The function is **MR_ambient_occlusion** and see the code for help with inputs and outputs.

## Dependencies

The MatlabRenderer must be downloaded and in your path for this code to run.

## Citation

This code was developed as part of the work described in the following paper. If you use this code in your research, please cite the paper.

William A. P. Smith, Alassane Seck, Hannah Dee, Bernard Tiddeman, Joshua Tenenbaum and Bernhard Egger. "A Morphable Face Albedo Model". In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Bibtex:

    @inproceedings{smith2020morphable,
      title={A Morphable Face Albedo Model},
      author={Smith, William A. P. and Seck, Alassane and Dee, Hannah and Tiddeman, Bernard and Tenenbaum, Joshua and Egger, Bernhard},
      booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2020}
    }
