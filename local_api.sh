bash
conda activate maskrcnn
gunicorn --bind 0.0.0.0:5000 wsgi --daemon
