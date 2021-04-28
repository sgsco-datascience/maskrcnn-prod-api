source /home/sgscolabs/anaconda3/etc/profile.d/conda.sh
conda activate maskrcnn
gunicorn --bind 0.0.0.0:5000 wsgi --timeout 1800 --daemon --error-logfile ./error.log --access-logfile ./access.log --capture-output --log-level debug


