FROM ubuntu:16.04

RUN apt-get update\
     && apt-get install -y curl\
     && apt-get install -y wget\
     && apt-get install -y git

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -c anaconda -y python=3.6
RUN conda install -c anaconda -y \
    pip \
    tensorflow-gpu=1.15.0\
    keras=2.1.6\
    numpy=1.16.0

RUN git clone https://WeinChien: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@github.com/sgsco-labs/Mask_RCNN
WORKDIR "/Mask_RCNN"
RUN pip install -r requirements.txt
RUN python setup.py build
RUN python setup.py install
WORKDIR "../"

#WORKDIR "/weights"
#RUN wget https://www.dropbox.com/s/wqx77t2q2r8uyoo/barcode_maskrcnn.h5?dl=1
#RUN git clone https://github.com/sgsco-labs/maskrcnn
RUN git clone https://WeinChien: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@github.com/sgsco-labs/maskrcnn.git
WORKDIR "/maskrcnn"
RUN wget https://www.dropbox.com/s/wqx77t2q2r8uyoo/barcode_maskrcnn.h5
RUN pip install -r requirements.txt

EXPOSE 9090
CMD ["python3", "flask_app.py"]