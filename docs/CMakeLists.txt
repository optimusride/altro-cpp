include(FindDoxygen)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_EXTRACT_STATIC YES)
doxygen_add_docs(doxygen 
  ${AltroCpp_SOURCE_DIR}/altro
  ${AltroCpp_SOURCE_DIR}/examples
  ${AltroCpp_SOURCE_DIR}/docs
)