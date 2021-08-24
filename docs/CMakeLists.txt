include(FindDoxygen)
set(DOXYGEN_USE_MATHJAX YES)
doxygen_add_docs(doxygen 
  ${altro_SOURCE_DIR}/altro
  ${altro_SOURCE_DIR}/examples
  ${altro_SOURCE_DIR}/docs
)