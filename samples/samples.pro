TEMPLATE = subdirs
SUBDIRS = host

equals(FLO_COMPILE_DEVICE_CODE, 1) {
    SUBDIRS += device
}
