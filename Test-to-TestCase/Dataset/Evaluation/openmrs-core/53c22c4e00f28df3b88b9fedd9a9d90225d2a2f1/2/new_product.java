@Override
	public boolean equals(Object obj) {
		if (obj instanceof ConceptComplex) {
			ConceptComplex c = (ConceptComplex) obj;
			return (this.getConceptId().equals(c.getConceptId()));
		} else if (obj instanceof Concept) {
			// use the reverse .equals in case we have hibernate proxies - #1511
			return OpenmrsUtil.nullSafeEquals(((Concept) obj).getConceptId(), this.getConceptId());
		}
		
		// fall back to object equality
		return obj == this;
	}