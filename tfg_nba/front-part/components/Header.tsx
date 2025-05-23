import { FunctionComponent } from "preact/src/index.d.ts";

const Header: FunctionComponent = () => {
    return (
        <div class="Header">
            <div><a href="/">Home</a></div>
            <div><a href="/ID">Solo equipos</a></div>
            <div><a href="/Stats">Laboratorio</a></div>
        </div>
    )
}

export default Header;