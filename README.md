# Project Documentation

## Prerequisites

Ensure the following are installed:
* **Docker Desktop** (includes Docker Compose)

## Installation and Setup

Follow these steps to run the application using the Docker Compose workflow.

### 1. Download

1.  Click the **Code** button on the repository page.
2.  Select **Download ZIP**.
3.  Extract (unzip) the contents to a folder.

### 2. Prepare Environment

1.  Open your terminal (CMD, PowerShell, or Bash).
2.  Copy the path of the extracted folder from your file manager.
3.  Navigate to the directory:

    ```bash
    cd <paste_path_here>
    # Example: cd C:\Users\Admin\Downloads\project-main
    ```

### 3. Run Application

Execute the build and start command. This creates the environment and starts the services.

```bash
docker compose up --build
