#!/usr/bin/env python  # noqa: EXE001, D100, CPY001

from bincrafters import build_template_default

if __name__ == '__main__':
    builder = build_template_default.get_builder(
        build_types=['Release'], archs=['x86_64']
    )
    builder.run()
