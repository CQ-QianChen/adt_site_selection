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

def engineName(surrogate_config, training_sampling_config, validation_sampling_config) {
    // surrogate id: its own settings plus the training and validation samples it sees
    def surrogate_hash8 = computeHash8([
        surrogate: surrogate_config,
        training: training_sampling_config,
        validation: validation_sampling_config
    ])
    return "engine_${surrogate_config.type.toLowerCase()}_${surrogate_hash8}"
}

def checkKeys(config, defaults, block) {
    def unknown = config.keySet() - defaults.keySet()
    if( unknown )
        error "unknown keys in the ${block} block: ${unknown.sort()}"
}
