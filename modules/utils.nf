def toYaml(data) {
    def yb = new groovy.yaml.YamlBuilder()
    yb.call(data)
    return yb.toString()
}

def canonicalize(data) {
    if (data instanceof Map) {
        return new TreeMap(data.collectEntries { k, v -> [(k): canonicalize(v)] })
    } else if (data instanceof List) {
        return data.collect { canonicalize(it) }
    }
    return data
}

def computeHash8(data) {
    def json = groovy.json.JsonOutput.toJson(canonicalize(data))
    def digest = java.security.MessageDigest.getInstance("SHA-256").digest(json.getBytes("UTF-8"))
    def hex = digest.collect { String.format("%02x", it) }.join()
    return hex[-8..-1]
}

// the part of the config that identifies the case (and so the datastore folder)
def caseConfig(config) {
    return config.findAll { k, v -> !(k in ['simulator_config', 'surrogate', 'uq']) }
}

def caseHash(config) {
    return computeHash8(caseConfig(config))
}

def engineName(surrogate_config, training_sampling_config, validation_sampling_config, scenario_hash8) {
    // surrogate id: the scenario, its own settings plus the training and validation samples it sees
    def surrogate_hash8 = computeHash8([
        scenario: scenario_hash8,
        surrogate: surrogate_config,
        training: training_sampling_config,
        validation: validation_sampling_config
    ])
    return "engine_${surrogate_config.type.toLowerCase()}_${surrogate_hash8}"
}

// the engine to use: the one named in uq.engine, else the one the scenario and surrogate settings give (it may not exist yet)
def resolveEngine(case_config, case_name, surrogate_config, sampling_config, project_dir) {
    def hash8 = caseHash(case_config)
    def models_dir = file("${project_dir}/datastore/${case_name}/${hash8}/surrogate_models")
    def engine = case_config.uq.engine
    if( engine == null )
        engine = engineName(surrogate_config, sampling_config.training, sampling_config.validation, hash8)
    else if( !(engine instanceof CharSequence) )
        error "uq.engine must be an engine name or null, got: ${engine}"
    else if( !file("${models_dir}/${engine}/engine.h5").exists() ) {
        def found = files("${models_dir}/*/engine.h5").collect { it.parent.name }.sort()
        error "uq.engine ${engine} not found in ${models_dir}; available: ${found}"
    }
    return [engine, file("${models_dir}/${engine}")]
}

def checkKeys(config, defaults, block) {
    def unknown = config.keySet() - defaults.keySet()
    if( unknown )
        error "unknown keys in the ${block} block: ${unknown.sort()}"
}

def mergeSampling(sampling, defaults) {
    if( sampling?.keySet() != ['training', 'validation'] as Set )
        error "surrogate.sampling needs training and validation, got: ${sampling?.keySet()}"
    def required = ['sample_size', 'sampling_method', 'seed']
    return sampling.collectEntries { name, entry ->
        def missing = required - entry.keySet()
        if( missing )
            error "surrogate.sampling.${name} is missing: ${missing}"
        [(name): defaults + entry]
    }
}

// the yaml with the defaults filled in, the surrogate settings without their sampling, and the sampling config
def mergeDefaults(config, defaults, case_name) {
    config.communication_sdh = defaults.communication_sdh + config.communication_sdh
    config.model = defaults.model + config.model + [project_name: case_name]
    config.simulator_config = defaults.simulator_config + config.simulator_config
    checkKeys(config.uq ?: [:], defaults.uq, 'uq')
    config.uq = defaults.uq + (config.uq ?: [:])
    def sampling_config = mergeSampling(config.surrogate.sampling, defaults.sampling)
    config.surrogate = defaults.surrogate + config.surrogate + [
        pce_opts: defaults.surrogate.pce_opts + (config.surrogate.pce_opts ?: [:]),
        gp_opts: defaults.surrogate.gp_opts + (config.surrogate.gp_opts ?: [:])
    ]
    def surrogate_config = config.surrogate.findAll { k, v -> k != 'sampling' }
    return [config, surrogate_config, sampling_config]
}
