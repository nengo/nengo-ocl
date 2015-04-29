version_info = (0, 1, 0)  # (major, minor, patch)
rc = None
dev = True

version = "{0}{1}{2}".format('.'.join(str(v) for v in version_info),
                             '-rc{0:d}'.format(rc) if rc is not None else '',
                             '-dev' if dev else '')
