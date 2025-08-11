import os
import shlex
import subprocess
import sys
from scripts.python_helpers.start_api_server import _get_arg_parser

def main():
    if '--' in sys.argv:
        sep_index = sys.argv.index('--')
        app_argv = sys.argv[1:sep_index]
        gunicorn_argv = sys.argv[sep_index + 1 :]
    else:
        app_argv = sys.argv[1:]
        gunicorn_argv = [
            'gunicorn', '-c', 'gunicorn.conf.py',
            'scripts.python_helpers.start_api_server:create_app()'
        ]

    parser = _get_arg_parser()
    parser.parse_args(app_argv)

    # Set APP_ARGS so create_app() can parse them without interfering with gunicorn argv
    os.environ['APP_ARGS'] = ' '.join(shlex.quote(a) for a in app_argv)

    print('Running:', ' '.join(gunicorn_argv))
    sys.exit(subprocess.call(gunicorn_argv))

if __name__ == '__main__':
    main()
