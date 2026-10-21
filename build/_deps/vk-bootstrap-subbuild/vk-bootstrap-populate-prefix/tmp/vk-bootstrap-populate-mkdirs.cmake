# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-src")
  file(MAKE_DIRECTORY "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-build"
  "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-subbuild/vk-bootstrap-populate-prefix"
  "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-subbuild/vk-bootstrap-populate-prefix/tmp"
  "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-subbuild/vk-bootstrap-populate-prefix/src/vk-bootstrap-populate-stamp"
  "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-subbuild/vk-bootstrap-populate-prefix/src"
  "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-subbuild/vk-bootstrap-populate-prefix/src/vk-bootstrap-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-subbuild/vk-bootstrap-populate-prefix/src/vk-bootstrap-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/slavr/OneDrive/Documents/CG_Lab/build/_deps/vk-bootstrap-subbuild/vk-bootstrap-populate-prefix/src/vk-bootstrap-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
