import sys

from app.helpers.runtime_env import configure_qt_environment, detect_runtime_environment, print_runtime_diagnostics

configure_qt_environment()

from app.ui import main_ui
from PySide6 import QtWidgets

import qdarktheme
from app.ui.core.proxy_style import ProxyStyle

if __name__ == "__main__":
    print_runtime_diagnostics(detect_runtime_environment())
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle())
    with open("app/ui/styles/true_dark_styles.qss", "r") as f:
        _style = f.read()
        _style = (
            qdarktheme.load_stylesheet(
                theme="dark", custom_colors={"primary": "#4090a3"}
            )
            + "\n"
            + _style
        )
        app.setStyleSheet(_style)
    window = main_ui.MainWindow()
    window.show()
    app.exec()
