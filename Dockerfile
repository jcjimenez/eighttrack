FROM jcjimenez/opencv-docker:3.4-contrib-py3-cpu

WORKDIR /libspatialindex
RUN curl -L http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz | tar xz && \
    cd spatialindex-src-1.8.5 && \
    ./configure && \
    make && \
    make install

WORKDIR /eighttrack
ADD . /eighttrack
RUN python3 setup.py test && \
    python3 setup.py install && \
    cd / && \
    python3 -c 'import eighttrack' && \
    cd /eighttrack
