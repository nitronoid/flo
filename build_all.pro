TEMPLATE = subdirs
SUBDIRS = flo samples

#SUBDIRS += test

samples.depends = flo
test.depends = flo

