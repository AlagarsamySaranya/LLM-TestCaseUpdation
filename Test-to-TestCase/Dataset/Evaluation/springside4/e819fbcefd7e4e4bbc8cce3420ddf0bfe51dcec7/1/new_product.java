@Override
	public IdResponse createUser(UserDTO user) {
		try {
			User userEntity = BeanMapper.map(user, User.class);

			Long userId = accountManager.saveUser(userEntity);

			return new IdResponse(userId);
		} catch (ConstraintViolationException e) {
			String message = StringUtils.join(BeanValidators.extractPropertyAndMessage(e), "\n");
			return new IdResponse().setError(WSResponse.PARAMETER_ERROR, message);
		} catch (DataIntegrityViolationException e) {
			String message = "新建用户参数存在唯一性冲突(用户:" + user + ")";
			logger.error(message, e);
			return new IdResponse().setError(WSResponse.PARAMETER_ERROR, message);
		} catch (RuntimeException e) {
			logger.error(e.getMessage(), e);
			return new IdResponse().setDefaultError();
		}
	}