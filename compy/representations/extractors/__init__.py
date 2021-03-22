from typing import Optional
import contextlib

from .extractors import ClangDriver

@contextlib.contextmanager
def clang_driver_scoped_options(clang_driver, additional_include_dir: Optional[str] = None, filename: Optional[str] = None):
    """A context manager to set and restore options for the clang driver in a local scope.

    >>> with clang_driver_scoped_options(clang_driver, filename="foo"):
    ...     # clang_driver's file name is set to foo in this scope
    ...     pass
    ... # filename is restored to previous value after the with block
    """
    prev_filename = None
    if filename is not None:
        prev_filename = clang_driver.getFileName()
        clang_driver.setFileName(filename)
    
    if additional_include_dir:
        clang_driver.addIncludeDir(
            additional_include_dir, ClangDriver.IncludeDirType.User
        )
    
    try:
        yield clang_driver
    finally:
        if prev_filename is not None:
            clang_driver.setFileName(prev_filename)
        if additional_include_dir:
            clang_driver.removeIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )
               
