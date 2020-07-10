// window.addEventListener('load',() => {
//     const prel = document.querySelector('.preload');
//     prel.classList.add("preload-finish");
// });

const loader = document.querySelector('.preload')
const main = document.querySelector('.main')
main.style.display='none';
main.style.opacity=0;
function init(){
    setTimeout(() => {
        // loader.style.opacity=0;
        // loader.style.display='none';
        // main.style.display='block';
        setTimeout(() => (main.style.opacity=1),50);
        var s=1;
    },4000);
    window.addEventListener('load',() => {

        // loader.classList.add('.preload-finish');
        // main.style.display='block';
        setTimeout(() => (main.style.opacity=1),500);

        loader.style.opacity=0;
        loader.style.display='none';
        main.style.display='block';
        setTimeout(() => (main.style.opacity=1),50);
        // main.style.opacity=1;

    }
    );
}
init();