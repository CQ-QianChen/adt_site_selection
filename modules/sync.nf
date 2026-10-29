// runs on the machine itself: sync.py needs only python3, and the token (sync/.token or $ZENODO_TOKEN) stays out of the images
process SYNC {
    cache false
    debug true

    input:
    val datastore
    val record
    val url

    script:
    """
    ZENODO_URL=${url} python3 ${projectDir}/sync/sync.py ${datastore} ${record ? "--record ${record}" : ''}
    """
}
