#!/bin/bash

# Audio Analysis Pipeline Setup Script
# This script sets up the complete environment for Hindi ASR/TTS quality assessment

set -e  # Exit on any error

# Check if we're in a terminal that supports colors
if [ -t 1 ]; then
    # Terminal supports colors
    HAS_COLORS=true
else
    # No terminal colors (e.g., running in CI)
    HAS_COLORS=false
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="audio-analysis"
ENV_FILE="environment.yml"
PYTHON_VERSION="3.10"

# Function to print colored output
print_status() {
    if [ "$HAS_COLORS" = true ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    else
        echo "[INFO] $1"
    fi
}

print_success() {
    if [ "$HAS_COLORS" = true ]; then
        echo -e "${GREEN}[SUCCESS]${NC} $1"
    else
        echo "[SUCCESS] $1"
    fi
}

print_warning() {
    if [ "$HAS_COLORS" = true ]; then
        echo -e "${YELLOW}[WARNING]${NC} $1"
    else
        echo "[WARNING] $1"
    fi
}

print_error() {
    if [ "$HAS_COLORS" = true ]; then
        echo -e "${RED}[ERROR]${NC} $1"
    else
        echo "[ERROR] $1"
    fi
}

# Progress bar functions
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    printf "\r["
    if [ "$HAS_COLORS" = true ]; then
        printf "${GREEN}"
    fi
    printf "%${filled}s" | tr ' ' '█'
    if [ "$HAS_COLORS" = true ]; then
        printf "${NC}"
    fi
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%%" $percentage
    
    if [ $current -eq $total ]; then
        echo ""
    fi
}

# Spinner animation
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "\r[%c] " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    printf "\r    \r"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check conda installation
check_conda() {
    if ! command_exists conda; then
        print_error "Conda is not installed or not in PATH"
        echo ""
        echo "Please install conda first:"
        echo "1. Download Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        echo "2. Install it and restart your terminal"
        echo "3. Run this script again"
        exit 1
    fi
    
    print_success "Conda found: $(conda --version)"
}

# Function to check if environment already exists
check_existing_env() {
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Environment '${ENV_NAME}' already exists"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing environment..."
            conda env remove -n "${ENV_NAME}" -y
            print_success "Existing environment removed"
        else
            print_status "Using existing environment"
            return 0
        fi
    fi
}

# Function to create environment
create_environment() {
    print_status "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file '${ENV_FILE}' not found"
        exit 1
    fi
    
    echo ""
    print_status "Step 1/4: Creating conda environment..."
    show_progress 1 4
    
    # Start conda environment creation in background
    conda env create -f "$ENV_FILE" > /tmp/conda_create.log 2>&1 &
    local conda_pid=$!
    
    # Show spinner while conda is working
    spinner $conda_pid
    
    # Wait for conda to finish
    wait $conda_pid
    local conda_exit_code=$?
    
    show_progress 2 4
    
    if [ $conda_exit_code -eq 0 ]; then
        print_success "Environment created successfully"
    else
        print_error "Failed to create environment"
        echo "Conda output:"
        cat /tmp/conda_create.log
        exit 1
    fi
}

# Function to verify environment
verify_environment() {
    print_status "Verifying environment installation..."
    
    # Activate environment and test imports
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    
    show_progress 3 4
    
    # Test Python version
    print_status "Step 2/4: Checking Python version..."
    PYTHON_VER=$(python --version 2>&1 | cut -d' ' -f2)
    if [[ "$PYTHON_VER" == "${PYTHON_VERSION}"* ]]; then
        print_success "Python version: $PYTHON_VER"
    else
        print_error "Python version mismatch. Expected ${PYTHON_VERSION}*, got $PYTHON_VER"
        exit 1
    fi
    
    # Test key package imports
    print_status "Step 3/4: Testing package imports..."
    
    local import_errors=0
    local total_packages=7
    local current_package=0
    
    # Array of packages to test
    declare -a packages=(
        "torch:PyTorch"
        "librosa:Librosa"
        "bert_score:BERT Score"
        "transformers:Transformers"
        "webrtcvad:WebRTC VAD"
        "indicnlp:Indic NLP"
        "rich:Rich"
    )
    
    for package_info in "${packages[@]}"; do
        IFS=':' read -r import_name display_name <<< "$package_info"
        current_package=$((current_package + 1))
        
        printf "\rTesting %s... " "$display_name"
        
        if python -c "import $import_name; print('OK')" 2>/dev/null; then
            if [ "$HAS_COLORS" = true ]; then
                printf "${GREEN}✓${NC}\n"
            else
                printf "✓\n"
            fi
        else
            if [ "$HAS_COLORS" = true ]; then
                printf "${RED}✗${NC}\n"
            else
                printf "✗\n"
            fi
            print_error "$display_name import failed"
            import_errors=$((import_errors + 1))
        fi
    done
    
    if [ $import_errors -eq 0 ]; then
        print_success "All packages imported successfully"
    else
        print_error "$import_errors package(s) failed to import"
        exit 1
    fi
}

# Function to test the analyzer
test_analyzer() {
    print_status "Step 4/4: Testing analyzer script..."
    
    if [ ! -f "analyze.py" ]; then
        print_warning "analyze.py not found - skipping analyzer test"
        show_progress 4 4
        return 0
    fi
    
    # Test if analyzer can be imported and help works
    if python analyze.py --help >/dev/null 2>&1; then
        print_success "Analyzer script is working"
    else
        print_error "Analyzer script test failed"
        exit 1
    fi
    
    show_progress 4 4
}

# Function to display usage instructions
show_usage() {
    echo ""
    echo "To use the audio analysis pipeline:"
    echo ""
    echo "1. Activate the environment:"
    echo "   conda activate ${ENV_NAME}"
    echo ""
    echo "2. Run the analysis:"
    echo "   bash run_analyzer.sh"
    echo ""
    echo "3. Or run manually:"
    echo "   python analyze.py --reference_json your_data.json --output_json results.jsonl --output_csv stats.csv --use_gpu"
    echo ""
    echo "4. To deactivate the environment:"
    echo "   conda deactivate"
    echo ""
    echo "Environment name: ${ENV_NAME}"
    echo "Python version: ${PYTHON_VERSION}"
    echo ""
}

# Main setup process
main() {
    echo "=========================================="
    echo "Audio Analysis Pipeline Setup"
    echo "=========================================="
    echo ""
    
    # Check conda installation
    check_conda
    
    # Check for existing environment
    check_existing_env
    
    # Create environment
    create_environment
    
    # Verify installation
    verify_environment
    
    # Test analyzer
    test_analyzer
    
    echo ""
    print_success "Setup completed successfully!"
    
    # Show usage instructions
    show_usage
}

# Run main function
main "$@" 
