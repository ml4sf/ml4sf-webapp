$("#theForm").submit(function (event) {
    event.preventDefault();
});

$("#btnProcess").on('click', function (e) {
    var self = this;
    $.ajax({
        type: "POST",
        url: "/process",
        data: new FormData(document.getElementById('theForm')),
        cache: false,
        contentType: false,
        processData: false,
        beforeSend: () => {
            $(self).html(
                '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>'
            );
            $(self).prop("disabled", true);
        }
    })
    .done((data) => $("#results").html(data))

    .fail(() => $("#results").html("Please try again"))

    .always(() => {
        $(self).html('Process');
        $(self).removeAttr('disabled');
    });
});
