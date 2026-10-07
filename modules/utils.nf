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
