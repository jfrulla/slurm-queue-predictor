export UV_INSTALL_DIR := /opt/uv/bin
export VENV_BIN := .venv/bin/

UV := $(UV_INSTALL_DIR)/uv

.PHONY: install
install: $(UV) 
	$(UV) sync

$(UV):
	curl -LsSf https://astral.sh/uv/install.sh | sh
