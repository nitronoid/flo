TEMPLATE = subdirs
SUBDIRS = flo samples test

samples.depends = flo
test.depends = flo

