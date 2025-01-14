# survey_tools

This repository provides a set of tools and scripts for planning and optimizing surveys of high-redshift galaxies.

## Getting Started

### Prerequisites

Ensure you have the following software installed:

- Python 3.13 or higher
- Required Python libraries (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nvnunes/survey_tools.git
   cd survey_tools
   ```
2. Install survey_tools locally as an editable package:
   ```bash
   pip3 install -e path/to/survey_tools
   ```
3. Optionally, download data files (see `data/README.txt`)

## Directory Structure

```
survey_tools/
│
├── aomap/                    # Map AO performance across celestial sphere
├── create/                   # Scripts for generating cross-matched catalogs
├── data/                     # Data used by some tools
│   ├── catalogs/             # Third-party and cross-matched spectroscopic catalogs
│   ├── images/               # Catalog images (where available)
│   └── sky/                  # Atmospheric transmission and background data
├── docs/                     # Documentation and references (under development)
├── examples/                 # Jupyter notebooks demonstrating tool usage
│   ├── healpix.ipynb         # Working with HEALpix
│   ├── read-catalog.ipynb    # Reading and processing a spectroscopic catalog
│   ├── read-DESI.ipynb       # Reading the DESI ELG catalog (Early Data Release)
│   ├── sky.ipynb             # Using atmospheric transmission and line rejection tools
│   └── stars.ipynb           # Loading star data and finding asterisms
├── survey_tools/             # Core source code for the tools
│   ├── asterism.py           # Identify asterisms in a star catalog
│   ├── catalog.py            # Read and consolidate various spectroscopic catalogs
│   ├── gaia.py               # Query and retrieve star data from the Gaia database
│   ├── healpix.py            # Convert between HEALpix and celestial coordinates
│   ├── match.py              # Cross-match catalogs by ID and celestial coordinates
│   └── sky.py                # Atmospheric transmission and emission line rejection
├── targets/                  # Select galaxy targets from catalogs
├── tests/                    # Unit tests (under development)
├── requirements.txt          # List of Python dependencies
└── README.md                 # Project overview, installation, and contents
```

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository
2. Create a new branch for your feature or bug-fix
3. Submit a pull request with a detailed description of your changes

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

This project is being developed as part of the GIRMOS project at York University. We are grateful to the open-source community for providing the frameworks and tools that make this work possible.

## Contact

For questions, feedback, or collaboration opportunities, please contact [nvnunes@yorku.ca](mailto:nvnunes@yorku.ca).
