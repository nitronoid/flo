TEMPLATE = subdirs
SUBDIRS = flo demo test

demo.depends = flo
test.depends = flo

