from juliacall import Main as jl


_capture_display = jl.seval("""
    function(x)
        mktemp() do path, file
            buffer = IOBuffer()
            pushdisplay(TextDisplay(buffer))

            try
                redirect_stdout(file) do
                    display(x)
                end
            finally
                popdisplay()
            end

            flush(file)
            seekstart(file)
            stdout_output = read(file, String)
            display_output = String(take!(buffer))

            return isempty(stdout_output) ? display_output : stdout_output
        end
    end
""")


def capture_julia_display(jl_obj) -> str:
    return str(_capture_display(jl_obj)).rstrip()
