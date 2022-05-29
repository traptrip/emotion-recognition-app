fetch("/tasks").then(function (response) {
    return response.json();
}).then(function (data) {
    console.log(data);
    for (let i = 0; i < data.length; i++) {
        let result_video_url = ``;
        let result_meta_url = ``;
        if (data[i]['result_video_url'] != null) {
            result_video_url = `<a href="/tasks/${data[i]['id']}/video">Result Video</a>`;
            result_meta_url = `<a href="/tasks/${data[i]['id']}/meta">Result Metadata</a>`;;
        }

        $(`#task-table > tbody:last-child`).append(`<tr>\n` +
            `            <td>${data[i]['id']}</td>\n` +
            `            <td>${data[i]['status']}</td>\n` +
            `            <td class="centred-td-style">\n` + result_video_url +
            `            </td>\n` +
            `            <td class="centred-td-style">\n` + result_meta_url +
            `            </td>\n` +
            `            <td class="centred-td-style">\n` +
            `                <svg onclick="deleteTask(this)" xmlns="http://www.w3.org/2000/svg"  viewBox="0 0 512 512" width="16px" height="16px"><rect style="fill:#E21B1B;" width="512" height="512"/><g><rect x="227.972" y="97.607" transform="matrix(0.7071 -0.7071 0.7071 0.7071 -106.0462 255.9763)" style="fill:#FFFFFF;" width="55.991" height="316.781"/><rect x="97.652" y="228.001" transform="matrix(0.7071 -0.7071 0.7071 0.7071 -106.0244 256.0289)" style="fill:#FFFFFF;" width="316.781" height="55.991"/></g></svg>\n` +
            `            </td>\n` +
            `        </tr>`);
    }
})


function deleteTask(element) {
    let task_id = element.parentElement.parentElement.childNodes[1].textContent
    $(element).closest("tr").remove();
    $.ajax({
        type: "DELETE",
        url: `/tasks/${task_id}`,
        data: null,
        dataType: null,
        processData: false,
        contentType: false,
    });
}