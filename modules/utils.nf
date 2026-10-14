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

def engineName(surrogate_config, training_sampling_config, validation_sampling_config) {
    // surrogate id: its own settings plus the training and validation samples it sees
    def surrogate_hash8 = computeHash8([
        surrogate: surrogate_config,
        training: training_sampling_config,
        validation: validation_sampling_config
    ])
    return "engine_${surrogate_config.type.toLowerCase()}_${surrogate_hash8}"
}

def checkEngine(engine, models_dir) {
    if( engine == null )
        return
    if( !(engine instanceof CharSequence) )
        error "uq.engine must be an engine name or null, got: ${engine}"
    if( !file("${models_dir}/${engine}/engine.h5").exists() ) {
        def found = files("${models_dir}/*/engine.h5").collect { it.parent.name }.sort()
        error "uq.engine ${engine} not found in ${models_dir}; available: ${found}"
    }
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
