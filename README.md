# MFICA Spectrum Decomposition

---

Github repository for code to decompose quasar and galaxy spectra using spectral components/templates generated using Mean-Field Independent Component Analysis (MF-ICA).

Included in this repository is an [example](https://github.com/SahyadriDK/MFICA/blob/main/example.ipynb) Jupyter notebook that demonstrates usage of MFICA spectrum decomposition.

MFICA spectrum decomposition takes arrays of wavelengths, fluxes and errors on fluxes for a given spectrum and decomposes this spectrum into its host galaxy and purely AGN spectra. From these spectra, MFICA can also return useful host galaxy properties, including estimates of the current stellar populations of the given object.

---

### Basic Installation:

---

### Citation:

If you find this code useful for your work, please cite [Krishna et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025MNRAS.543.1535K/abstract):

```latex
@ARTICLE{2025MNRAS.543.1535K,
       author = {{Krishna}, Sahyadri D. and {Wild}, Vivienne and {Hewett}, Paul C. and {Villforth}, Carolin},
        title = "{Stellar populations of quasar host galaxies with MFICA decomposition}",
      journal = {\mnras},
     keywords = {galaxies: evolution, quasars: general, galaxies: star formation, galaxies: active, Astrophysics of Galaxies},
         year = 2025,
        month = oct,
       volume = {543},
       number = {2},
        pages = {1535-1555},
          doi = {10.1093/mnras/staf1557},
archivePrefix = {arXiv},
       eprint = {2509.10271},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.543.1535K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

---

### Issues and Contact:

In case of any issues, or ideas for extending the functionality of this code and applying to new domains, send an email to [sk322@st-andrews.ac.uk](mailto:sk322@st-andrews.ac.uk).