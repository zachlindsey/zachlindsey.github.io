import livereload
server = livereload.Server()
server.watch('.')
server.serve(port=8080)
