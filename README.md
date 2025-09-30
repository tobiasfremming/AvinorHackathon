
# AvinorCompetition

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/tobiasfremming/AvinorHackathon/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/tobiasfremming/AvinorHackathon)
![GitHub language count](https://img.shields.io/github/languages/count/tobiasfremming/AvinorHackathon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)


</div>

<details> 
<summary><b>📋 Table of contents </b></summary>

- [AvinorCompetition](#AvinorCompetition)
  - [Description](#description)
  - [🛠️ Prerequisites](#%EF%B8%8F-prerequisites)
  - [Getting started](#getting-started)
  - [Usage](#usage)
    - [📖 Generate Documentation Site](#-generate-documentation-site)
  - [Testing](#testing)
  - [Team](#team)
    - [License](#license)

</details>

## Description

<!-- TODO: Provide a brief overview of what this project does and its key features. Please add pictures or videos of the application -->
This is our solution for Avinor’s data competition on the development of a concurrency model. The model predicts the probability that AFIS officers will need to communicate with multiple aircraft simultaneously, with a focus on understanding workload in multiple-operations scenarios.

The task requires the development of a concurrency model that predicts the probability of simultaneous dialogue between AFIS officers and multiple aircraft. The main challenge does not lie in the planned flight schedules, which Avinor already has access to, but in predicting deviations from scheduled times and translating these into concurrency probabilities. Our approach focuses on modeling delays and irregularities that affect actual air traffic, as well as weather-related and operational factors that increase the need for communication.

## 🛠️ Prerequisites

<!-- TODO: In this section you put what is needed for the program to run.
For example: OS version, programs, libraries, etc.  

-->

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

## Getting started

<!-- TODO: In this Section you describe how to install this project in its intended environment.(i.e. how to get it to run)  
-->

1. **Clone the repository**:

   ```sh
   git clone https://github.com/CogitoNTNU/AvinorCompetition.git
   cd AvinorCompetition
   ```

1. **Install dependencies**:

   ```sh
   uv sync
   ```

<!--
1. **Configure environment variables**:
    This project uses environment variables for configuration. Copy the example environment file to create your own:
    ```sh
    cp .env.example .env
    ```
    Then edit the `.env` file to include your specific configuration settings.
-->

1. **Set up pre commit** (only for development):
   ```sh
   uv run pre-commit install
   ```

## Usage

To run the project, run the following command from the root directory of the project:

```bash
avinorhackathon

```

<!-- TODO: Instructions on how to run the project and use its features. -->

### 📖 Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference. Get the documentation according to the lastes commit on main by viewing the `gh-pages` branch on GitHub: [https://cogitontnu.github.io/AvinorCompetition/](https://cogitontnu.github.io/AvinorCompetition/).

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```

## Team

This project would not have been possible without the hard work and dedication of all of the contributors. Thank you for the time and effort you have put into making this project a reality.



### License

______________________________________________________________________

Distributed under the MIT License. See `LICENSE` for more information.
